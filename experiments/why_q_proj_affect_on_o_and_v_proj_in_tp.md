`o_proj` меняется потому, что Torch-Pruning ошибочно трактует весь путь через attention как сохранение одной и той же «канальной» оси. Для `q_proj.out` это математически неверно: размерность Q сворачивается в `Q @ Kᵀ` и не должна напрямую доходить до `o_proj`.

### Реальная математика attention

Пусть:

```text
Q:       [B, Hq, S, Dqk]
K:       [B, Hq, T, Dqk]
V:       [B, Hq, T, Dv]

Q @ Kᵀ: [B, Hq, S, T]      # Dqk исчезла
P @ V:   [B, Hq, S, Dv]    # последняя ось пришла от V
reshape: [B, S, Hq * Dv]
o_proj:  Linear(Hq * Dv, hidden_size)
```

Следовательно:

```text
q_proj.out[Dqk]
    └── должен связываться с k_proj.out[Dqk]

v_proj.out[Dv]
    └── должен связываться с o_proj.in[Hq * Dv]
```

Прямой зависимости

```text
q_proj.out → o_proj.in
```

математически нет.

### Как Torch-Pruning строит ошибочный путь

Для реализации TransformerLens путь получается таким:

```text
q_proj.out
  → view / transpose
  → RoPE: mul, rotate_half, mul, add
  → Q @ Kᵀ
  → scaling
  → mask
  → softmax
  → attention_weights @ V
  → transpose
  → reshape
  → o_proj.in
```

Это буквально соответствует:

- Q reshape: [position_embeddings_attention.py:375](/glazkov-dev/TransformerLens/transformer_lens/model_bridge/generalized_components/position_embeddings_attention.py:375)
- `Q @ Kᵀ`: [position_embeddings_attention.py:428](/glazkov-dev/TransformerLens/transformer_lens/model_bridge/generalized_components/position_embeddings_attention.py:428)
- softmax: [position_embeddings_attention.py:450](/glazkov-dev/TransformerLens/transformer_lens/model_bridge/generalized_components/position_embeddings_attention.py:450)
- `attention_weights @ V`: [position_embeddings_attention.py:464](/glazkov-dev/TransformerLens/transformer_lens/model_bridge/generalized_components/position_embeddings_attention.py:464)
- reshape и `o_proj`: [position_embeddings_attention.py:465](/glazkov-dev/TransformerLens/transformer_lens/model_bridge/generalized_components/position_embeddings_attention.py:465)

В обычном Hugging Face Llama внешний путь тот же:

- Q/K/V projection: [modeling_llama.py:262](/glazkov-dev/LoRa-Transfer-Pruning/.venv/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py:262)
- вызов attention backend: [modeling_llama.py:276](/glazkov-dev/LoRa-Transfer-Pruning/.venv/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py:276)
- reshape и `o_proj`: [modeling_llama.py:287](/glazkov-dev/LoRa-Transfer-Pruning/.venv/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py:287)

При `eager` внутри находятся два явных `matmul`; при `sdpa` они спрятаны в `scaled_dot_product_attention`.

### Почему TP не останавливается на matmul

В вашей версии TP практически все неизвестные autograd-операции попадают сюда:

```python
module = ops._ElementWiseOp(self._op_id, grad_fn.name())
```

Это видно в [graph.py:559](/glazkov-dev/LoRa-Transfer-Pruning/.venv/lib/python3.10/site-packages/torch_pruning/dependency/graph.py:559).

Поэтому:

```text
BmmBackward0
MmBackward0
SoftmaxBackward0
MaskedFillBackward0
CloneBackward0
TransposeBackward0
```

обрабатываются одинаковым `ElementWisePruner`, который просто пропускает индексы без понимания осей. В [ops.py](/glazkov-dev/LoRa-Transfer-Pruning/.venv/lib/python3.10/site-packages/torch_pruning/ops.py) он фактически пустой:

```python
class ElementWisePruner(DummyPruner):
    pass
```

А правила dependency graph соединяют `out pruning` текущей вершины с `in pruning` следующей вершины: [graph.py:403](/glazkov-dev/LoRa-Transfer-Pruning/.venv/lib/python3.10/site-packages/torch_pruning/dependency/graph.py:403).

Именно поэтому TP делает:

```text
q indices
 → те же indices через первый BMM
 → scores
 → те же indices через второй BMM
 → attention output
 → prune_linear_in_channels(o_proj, indices)
```

Последняя операция физически удаляет соответствующие столбцы:

```text
o_proj.weight:
[hidden_size, H * Dv]
             ↓
[hidden_size, H * Dv - len(indices)]
```

То есть увиденное вами:

```python
o_proj: in_features 4096 → 3648
weight: [4096, 4096] → [4096, 3648]
```

является результатом `prune_linear_in_channels`, вызванного ложной сквозной зависимостью.

### Отличие HF Llama от TransformerLens bridge

Принципиального отличия в default-пути нет. Bridge вызывает:

```python
query_states = self.q(hidden_states)
...
attn_output = self.o(attn_output)
```

Но `LinearBridge` оборачивает тот же самый HF `q_proj/o_proj`, поэтому DependencyGraph в итоге доходит до `_original_component`.

Ветка `use_attn_result=True` отличается: там bridge вручную использует `o_proj.weight` для per-head результата. Эту ветку TP может трассировать ещё менее корректно, потому что обычного вызова `o_proj(attn_output)` там нет.

### Что следует делать с группой

Если вы структурно уменьшаете только Q/K head dimension, из группы нужно удалить операции над:

- `v_proj.out`;
- `o_proj.in`;
- путём между вторым matmul и `o_proj`.

Оставить надо:

```text
q_proj.out
↔ соответствующие RoPE-пары
↔ k_proj.out с GQA mapping
```

Но текущая реализация Llama использует один `self.head_dim` одновременно для Q, K и V:

```python
hidden_shape = (*input_shape, -1, self.head_dim)
```

Поэтому отдельно уменьшить только `Dqk`, сохранив `Dv=128`, без изменения `forward` нельзя. Нужны раздельные:

```python
qk_head_dim = 114
v_head_dim = 128
```

и reshape:

```python
Q/K → qk_head_dim
V   → v_head_dim
```

Тогда `o_proj.in_features` остаётся `num_heads * v_head_dim`.

Если же вы хотите сохранить стандартную Llama-структуру с единым `head_dim=114`, тогда действительно необходимо прунить:

```text
Q: 32 × 128 → 32 × 114
K:  8 × 128 →  8 × 114
V:  8 × 128 →  8 × 114
O input: 32 × 128 → 32 × 114
```

Но индексы для `V → O` должны строиться отдельно по семантике голов, а не приходить из Q через softmax/BMM. Текущая группа TP случайно приходит к похожему набору модулей, но путь и mapping у неё неверные.