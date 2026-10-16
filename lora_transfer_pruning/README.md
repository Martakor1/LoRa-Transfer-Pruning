# LoRA Transfer Pruning

A package for simulating structural LLM pruning and training LoRA adapters for
different pruning configurations on a shared base model. Models are represented
through `TransformerLens.TransformerBridge`; channel dependencies are determined
using `torch_pruning.DependencyGraph` and architecture-specific adjustments.

Two workflows share the same pruning plan:

- **SP / transfer pruning:** hooks zero selected activations while preserving
  weight shapes. Masks can switch with the active LoRA adapter. This alone does
  not reduce model size or guarantee faster computation.
- **TP / structural pruning:** Torch-Pruning groups physically remove channels,
  while structural setup updates the associated dimensions and attention settings.

## Architecture and entities

```text
ModelPruneTask → LocalPruning → TorchPruningGroupBuilder → PruneTaskPlan
                                                            ├─ SP: mapper → activation hooks
                                                            └─ TP: group.prune() + structural setup
```

| Layer | Responsibility |
| --- | --- |
| [`core/`](core/) | Task definitions, activation instrumentation, and multi-adapter training. |
| [`adapter/torch_pruning/`](adapter/torch_pruning/) | DG construction, index mapping, architecture-specific groups, and structural setup. |
| [`usecase/`](usecase/) | Creating plans and applying them through the public `LocalPruning` API. |

Main entities:

- `GroupPruneTask`: a request for one root module. `cols` specifies input channels
  and `rows` specifies output channels, as an index list, a `float` ratio, or `None`.
  `kv_lora_idxs_deepseek` separately controls pruning of the MLA latent KV dimension
  (not the rank of a PEFT adapter).
- `ModelPruneTask`: a mapping of `bridge module name → GroupPruneTask` and a task
  name. For LoRA, the name must match the adapter name; the default is `"default"`.
- `TorchPruningGroupTask`: an internal task representation with index tensors.
- `PruneTaskPlan`: a `name`, concrete `groups`, and corresponding `structural_setups`.
  A plan is tied to the model instance used to build the DG.
- `PruningInstrumentor`: hooks for masking, optional rescaling, RMSNorm correction,
  and MLA scaling; conditional mask activation based on the active adapter.
- `TorchPruningGroupMapper`: maps indices from TP groups to bridge component hooks.

## Supported features

- Explicit channel indices and random selection by pruning ratio. Set
  `torch.manual_seed` before building a plan for reproducibility.
- Linear/MLP groups and specialized attention paths:
  - standard Q/K/V attention: repeating local indices across heads, RoPE pairs, and GQA;
  - Gemma 4: groups spanning related layers with shared KV;
  - DeepSeek V2: direct-Q MLA, separate NOPE/PE channels, and latent KV pruning.
- Multiple named plans for different LoRA adapters on a shared model.

For attention, the task root is `blocks.N.attn.q` with `rows`, or
`blocks.N.attn.q_proj` for DeepSeek direct-Q. These indices are local to a head;
there is no need to repeat them manually. For an ordinary Linear, indices refer
directly to its input/output channels. A group may affect other modules and layers,
not just the specified root.

## Example: simulated pruning without LoRA

This assumes that `bridge` has already been created and `example_input_ids`
contains tokens on the appropriate device. Build the DG with autograd enabled
and gradient paths available, not inside `torch.no_grad()` or `inference_mode()`.

```python
import torch

from lora_transfer_pruning.core.prune_task_type import GroupPruneTask, ModelPruneTask
from lora_transfer_pruning.usecase.local_pruning import LocalPruning

torch.manual_seed(42)
task = ModelPruneTask(data={
    "blocks.0.attn.q": GroupPruneTask(cols=None, rows=[2, 6]),
    "blocks.0.mlp.up_proj": GroupPruneTask(cols=None, rows=0.2),
})

local = LocalPruning(bridge, example_input_ids)
plan = local.get_torch_pruning_groups_and_structural_setups(task)

# Masks are not bound to an adapter; structural setup is not called for SP.
local.prepare_model_to_transfer_pruning_from_groups(plan.groups, rescale=False)
```

`rescale=True` (the default) additionally restores the activation's L2 norm after
masking. This changes the computation; use `rescale=False` when comparing against
ordinary TP. SP preparation also installs dynamic RMSNorm correction.

## Other use cases

**Structural pruning.** Create a separate model instance and build its own plan,
then apply the groups and structural setups:

```python
for group, setup in zip(plan.groups, plan.structural_setups):
    group.prune()
    setup()
```

Do not reuse groups from another model or apply a plan repeatedly to already
pruned weights. An independent SP/TP comparison requires identical initial
weights and identical selected indices.

**Multiple LoRA adapters.** Create the adapters in PEFT, pass `peft_model` to
`LocalPruning`, name the tasks after their adapters, and call:

```python
plans = local.get_full_transfer_pruning_plan(tasks)
local.prepare_model_to_transfer_pruning(plans, rescale=False)
```

`MultiAdapterPruningTrainer` extends `trl.SFTTrainer`: it activates each adapter
in turn and performs forward/backward passes on the same batch. The optimizer
step remains in the standard Trainer loop. Base weights are frozen when training
starts. The optimizer must include the parameters of all adapters being trained;
enable their gradients before creating it. The trainer may unfreeze adapters in
every layer where they are installed. This class does not implement separate
evaluation of all adapters in `evaluate`.

**Equivalence checks.** Activation/gradient collection and comparison live in the
sibling package [`pruning_lens`](../pruning_lens/).

## Limitations

- RMSNorm correction is approximate: it counts zeros in the first input vector
  and multiplies the output by `sqrt(retained_channels / original_channels)`.
  It assumes a shared channel mask; natural zeros, epsilon, and rounding may
  introduce discrepancies.
- SP and TP need not match bit-for-bit: GEMM shapes, summation order, and dtype
  affect results even with equivalent masks.
- Arbitrary attention, MoE, and quantized modules are not universally supported.
  Building a DG alone does not guarantee correct physical pruning.
  The Gemma shared-KV builder currently requires plain Linear modules, not LoRA wrappers.
- For Gemma, specify one root per shared-KV family; conflicting masks within
  a family are rejected.
- Repeated SP preparation adds hooks; manage their lifecycle or use a fresh
  model instance for a new experiment.

## Tests and experiments

From the repository root, in a configured environment:

```bash
python -m pytest tests/lora_transfer_pruning -q
```

Unit tests are in [`tests/lora_transfer_pruning`](../tests/lora_transfer_pruning/),
integration checks in [`test_integration`](../test_integration/), and
research notebooks in [`experiments/compare_tp_and_our`](../experiments/compare_tp_and_our/).
