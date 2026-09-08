`(6-10): 5 x Gemma4TextDecoderLayer(...)` — это только сокращённый вывод `repr` у PyTorch, а не реальная группировка или шаринг слоёв.

В коде каждый слой создаётся отдельно:

```python
self.layers = nn.ModuleList(
    [Gemma4TextDecoderLayer(config, layer_idx)
     for layer_idx in range(config.num_hidden_layers)]
)
```

PyTorch замечает несколько подряд идущих модулей с одинаковым текстовым представлением и печатает их как:

```text
(6-10): 5 x Gemma4TextDecoderLayer(...)
```

При этом:

```python
model.layers[6] is model.layers[7]  # False
```

И параметры тоже отдельные:

```python
model.layers[6].mlp.gate_proj.weight.data_ptr() \
    == model.layers[7].mlp.gate_proj.weight.data_ptr()  # обычно False
```

Важно: в `repr` не показываются обычные Python-поля вроде `layer_idx`, `is_kv_shared_layer`, `layer_type`. Поэтому два архитектурно различающихся по поведению слоя могут визуально выглядеть одинаково.

## Что означает `is_kv_shared_layer`

Gemma 4 умеет делать cross-layer KV sharing: последние `num_kv_shared_layers` слоёв не вычисляют собственные `K` и `V`, а используют KV из более раннего слоя того же типа.

Граница вычисляется так:

```python
first_kv_shared_layer_idx = (
    config.num_hidden_layers - config.num_kv_shared_layers
)

self.is_kv_shared_layer = (
    layer_idx >= first_kv_shared_layer_idx > 0
)
```

Например:

```text
num_hidden_layers = 12
num_kv_shared_layers = 5

обычные слои:       0 ... 6
KV-shared слои:     7 ... 11
```

Но Gemma 4 имеет два типа attention:

```text
sliding_attention
full_attention
```

Поэтому shared-слой ищет последний обычный слой именно своего типа:

```python
self.kv_shared_layer_index = (
    len(prev_layers)
    - 1
    - prev_layers[::-1].index(config.layer_types[layer_idx])
)
```

Условно:

```text
layer 5: full attention       ← источник full KV
layer 6: sliding attention    ← источник sliding KV
-------------------------------- начало sharing
layer 7: sliding attention    → берёт KV слоя 6
layer 8: sliding attention    → берёт KV слоя 6
layer 9: full attention       → берёт KV слоя 5
```

При этом каждый слой всё равно вычисляет собственный `Q`:

```python
query_states = self.q_proj(hidden_states)
```

Shared только `K` и `V`:

```python
key_states, value_states = (
    past_key_values.shared_layers[self.kv_shared_layer_index]
)
```

Таким образом, верхние слои задают разные запросы к одному и тому же сохранённому представлению предыдущего слоя.

## Зачем `store_full_length_kv`

Последний non-shared слой каждого типа назначается источником:

```python
self.store_full_length_kv = layer_idx == last_non_shared_layer_of_same_type
```

После вычисления его KV делается дополнительное сохранение:

```python
past_key_values.shared_layers[self.layer_idx] = (
    key_states,
    value_states,
)
```

Это отдельный словарь поверх стандартного HF cache:

```text
past_key_values
├── стандартный KV cache по layer_idx
└── shared_layers
    ├── индекс source sliding layer → его полные K,V
    └── индекс source full layer    → его полные K,V
```

Почему нельзя просто обратиться к стандартному cache другого слоя:

- разные реализации cache могут обрезать sliding-window состояние;
- обычный cache адресуется текущим `layer_idx`;
- shared-слоям нужен KV конкретного source-слоя;
- source KV иногда надо сохранить в полном виде, даже если сам source использует sliding attention;
- при model/device parallelism источник и потребитель могут оказаться на разных устройствах, поэтому есть:

```python
key_states = key_states.to(query_states.device)
value_states = value_states.to(query_states.device)
```

Название `store_full_length_kv` слегка сбивает с толку: оно не означает «этот слой делает full attention». Оно означает «сохрани результат этого source-слоя для последующего cross-layer reuse».

## Странность с `past_key_values is not None`

Шаринг фактически включается только при наличии cache:

```python
if self.is_kv_shared_layer and past_key_values is not None:
    # reuse
else:
    # вычислить собственные K/V
```

Следовательно:

- `use_cache=True`: shared-слои реально переиспользуют KV;
- `use_cache=False`: каждый слой вычисляет собственные KV;
- `k_proj` и иногда `v_proj` существуют даже у shared-слоёв, чтобы поддерживать путь без cache, обучение, checkpointing и совместимость с общим API.

Это заметное отличие от обычной Llama: поведение KV-sharing здесь завязано на специальное расширение cache, а не только на математическую структуру слоя.

## Другие фундаментальные отличия от Llama

### 1. Гибридный attention

Llama использует одинаковый causal full attention во всех слоях:

```python
causal_mask = create_causal_mask(...)
```

Gemma 4 чередует типы, по умолчанию примерно 5 sliding к 1 full:

```python
sliding_window_pattern = 6

[
    "sliding_attention"
    if (i + 1) % 6
    else "full_attention"
]
```

Для них отдельно строятся:

- разные маски;
- разные RoPE embeddings;
- потенциально разные размеры attention head;
- разные KV-источники.

### 2. Разный `head_dim`

В Llama:

```python
head_dim = config.head_dim
```

Во всех слоях одинаково.

В Gemma 4:

```python
head_dim = (
    config.global_head_dim
    if full_attention
    else config.head_dim
)
```

То есть global/full attention может работать с существенно более широкими головами.

### 3. Вариант `K == V`

Для global attention возможно:

```python
self.use_alternative_attention = (
    config.attention_k_eq_v and not self.is_sliding
)
```

Тогда `v_proj` вообще отсутствует:

```python
self.v_proj = None
value_states = key_states
```

В Llama `K` и `V` всегда имеют отдельные проекции.

### 4. QK/V normalization

Gemma 4 нормализует каждую голову:

```python
query_states = self.q_norm(query_states)
key_states = self.k_norm(key_states)
value_states = self.v_norm(value_states)
```

Причём `v_norm` используется без обучаемого scale:

```python
with_scale=False
```

В данной реализации Llama отдельного QK/V norm нет.

Из-за нормализации Gemma выставляет:

```python
self.scaling = 1.0
```

А Llama использует классическое:

```python
self.scaling = head_dim**-0.5
```

### 5. Другой residual/norm sandwich

Llama:

```text
x → pre-attention norm → attention → + residual
  → pre-MLP norm       → MLP       → + residual
```

Gemma 4:

```text
x → pre-attention norm → attention
  → post-attention norm → + residual
  → pre-FFN norm → MLP
  → post-FFN norm → + residual
```

То есть Gemma имеет дополнительные post-нормализации внутри residual-ветвей.

### 6. Double-wide MLP в KV-shared слоях

```python
use_double_wide_mlp = (
    config.use_double_wide_mlp and is_kv_shared_layer
)
```

Тогда:

```python
intermediate_size = config.intermediate_size * 2
```

Идея в том, что верхние слои экономят вычисления/память на K/V, но получают более широкую нелинейную часть.

Это также может объяснять границы групп в `repr`: там, где меняется ширина MLP или состав модулей, PyTorch начинает новую repr-группу.

### 7. Опциональный MoE-блок

Gemma 4 может параллельно dense MLP запускать mixture-of-experts:

```text
dense MLP ───────────────┐
                         + → объединение
router → top-k experts ──┘
```

У обычной `LlamaDecoderLayer` такого пути нет.

### 8. Per-layer inputs

Gemma 4 умеет подавать отдельный дополнительный input в каждый decoder layer:

```python
per_layer_inputs[:, :, i, :]
```

Затем слой делает gated injection:

```python
gate = act(per_layer_input_gate(hidden_states))
injection = projection(gate * per_layer_input)
hidden_states = residual + norm(injection)
```

Это особенно нужно мультимодальной части Gemma 4. В базовой Llama все слои получают только hidden state предыдущего слоя.

### 9. Масштабирование embeddings и layer output

Gemma масштабирует embeddings на:

```python
hidden_size**0.5
```

А в конце каждого decoder layer есть:

```python
hidden_states *= self.layer_scalar
```

`layer_scalar` зарегистрирован как buffer, то есть это часть состояния модели, но не обучаемый `Parameter`.

## Коротко

Llama — однородный стек независимых full-attention decoder layers:

```text
каждый слой: собственные Q, K, V + одинаковая схема attention
```

Gemma 4 — неоднородный стек:

```text
sliding/full attention
+ разные RoPE/head_dim
+ QK/V norm
+ возможный K=V
+ cross-layer KV sharing
+ widened MLP
+ optional MoE
+ per-layer multimodal inputs
```

А `(6-10): 5 x` относится исключительно к красивой печати модели и не означает, что эти пять слоёв являются одним модулем или имеют общие веса.

# O MoE в gemma и почему он не может быть MoEBridge
`experts` стал `GeneralizedComponent` не из-за ошибки создания твоей модели. Это явно задано в Gemma 4 adapter:

```python
# MoE branch — present only when enable_moe_block (26B-A4B).
"router": GeneralizedComponent(name="router", optional=True),
"experts": GeneralizedComponent(name="experts", optional=True),
```

Поэтому в notebook закономерно появляется:

```text
blocks.0.experts: GeneralizedComponent
```

## Почему здесь не `MoEBridge`

У Gemma 4 MoE не заменяет стандартный MLP. В decoder layer есть параллельные ветви:

```text
dense mlp ─────────────┐
                      ├─ add
router → experts ──────┘
```

Причём HF разделяет MoE на два независимых модуля:

```python
self.router = Gemma4TextRouter(config)
self.experts = Gemma4TextExperts(config)
```

И вызов выглядит так:

```python
_, top_k_weights, top_k_index = self.router(hidden_states_flat)

hidden_states_2 = self.experts(
    hidden_states_2,
    top_k_index,
    top_k_weights,
)
```

`Gemma4TextExperts` не выполняет routing и не возвращает router scores. Он получает уже готовые:

```python
hidden_states
top_k_index
top_k_weights
```

`MoEBridge` рассчитан на другую форму компонента: единый MoE-модуль получает hidden states, сам маршрутизирует их и часто возвращает:

```python
(hidden_states, router_scores)
```

Это видно из его контракта:

```python
output = self.original_component(*args, **kwargs)

if isinstance(output, tuple):
    hidden_states = output[0]
    router_scores = output[1]
    self.hook_router_scores(router_scores)
```

У `Gemma4TextExperts.forward()` результат — один tensor, а router вообще находится sibling-модулем. Поэтому применение существующего `MoEBridge` только к `experts` не добавило бы корректной MoE-абстракции. Оно дало бы почти то же делегирование, что и `GeneralizedComponent`, плюс фактически бесполезный `hook_router_scores`.

## Что именно TransformerLens заявляет про Gemma 4

Локальный adapter прямо документирует поддержку вариантов:

- E2B/E4B с KV sharing и per-layer embeddings;
- 31B/26B-A4B с `K==V`;
- 26B-A4B с параллельной MoE-ветвью;
- MoE-модули отображаются как optional `router` и `experts`;
- вычисления делегируются HF для parity;
- processed/compatibility phase 3 и fold LN отключены из-за PLE/MoE residual topology.

В официальной документации Gemma 4 adapter указан как поддерживаемый для bridge phases `1, 2, 4`, но не как канонически преобразованный `MoEBridge`. [Документация Gemma4ArchitectureAdapter](https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.model_bridge.supported_architectures.html)

Общая документация описывает `MoEBridge` как замену MLP для архитектур вроде Mixtral, GraniteMoE и OLMoE. Это другая топология, чем Gemma 4 с dense MLP плюс отдельной параллельной MoE-ветвью. [TransformerLens adapter specification](https://transformerlensorg.github.io/TransformerLens/content/adapter_development/adapter-specification.html)

## Что поддерживается практически

Для Gemma 4 сейчас доступны:

```python
bridge.blocks[i].router.hook_in
bridge.blocks[i].router.hook_out

bridge.blocks[i].experts.hook_in
bridge.blocks[i].experts.hook_out
```

Но `experts.hook_in` перехватывает только первый tensor-аргумент — `hidden_states`. Дополнительные:

```python
top_k_index
top_k_weights
```

передаются дальше без отдельных HookPoint.

Также нет стандартного:

```python
bridge.blocks[i].experts.hook_router_scores
```

потому что scores принадлежат `router`, а не `experts`.


## Для твоего pruning use case

`GeneralizedComponent` скрывает внутренние параметры экспертов:

```python
gate_up_proj: [E, 2I, D]
down_proj:    [E, D, I]
```

Они являются непосредственно `nn.Parameter`, а не `nn.Linear`. Поэтому текущая логика mapper, ориентированная на `LinearBridge`, не сможет удобно прунить отдельных экспертов или промежуточные expert channels.

Если нужна полноценная поддержка pruning, логичнее создать специализированный `Gemma4ExpertsBridge`, который предоставляет:

```text
hook_in
hook_gate
hook_up
hook_activation
hook_down
hook_out
```

и знает размерности `[experts, intermediate, hidden]`. Просто заменить:

```python
GeneralizedComponent(name="experts")
```

на:

```python
MoEBridge(name="experts")
```

недостаточно: текущий `MoEBridge` не декомпозирует batched expert weights и не перехватывает `top_k_index/top_k_weights`.

Итого: TransformerLens поддерживает выполнение и базовые hooks Gemma 4 MoE через HF-delegated components, но не представляет её `experts` через канонический `MoEBridge`. Это осознанное ограничение текущего adapter mapping, вызванное отличающейся параллельной и раздельной router/experts топологией.