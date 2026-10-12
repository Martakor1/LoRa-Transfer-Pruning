# Pruning Lens

Pruning Lens compares activations and gradients produced by transfer pruning (SP, channel masking) and structural Torch-Pruning (TP). It helps locate differences at individual hook points and distinguish numerical differences from missing gradients, unknown coordinate mappings, and tensor restoration errors.

The package separates model preparation, trace capture, coordinate restoration, numerical comparison, and reporting. Architecture-specific pruning logic remains in the existing builders and structural setups; the comparator operates on captured traces.

## Usage overview

Use `ComparePrunedModelsUsecase` to capture and compare SP and TP traces with a shared pruning plan. Use `PruneAndCaptureUsecase` to capture either trace separately, then pass the traces to `GreatComparator.compare_traces()`.

The combined use case treats TP as the reference and SP as the candidate. Differences are computed as `candidate - reference`. Activations, gradients, and the scalar metric (currently model loss) are compared separately.

- Pass the actual TP groups when comparing a structurally pruned model. They provide the channel-removal history required for restoration.
- With `names_filter=None`, the use case selects affected linear boundaries from the groups, including dependencies.
- An explicit hook filter can capture additional points, but their pruning mappings may be unresolved.
- Set `compute_gradient=False` to skip backward. Gradient comparisons remain empty.
- Set `store_difference=False` to omit full difference tensors while still computing metrics.
- Inspect `CompairingTrace` programmatically or call `print_summary()` to display the computed results.

**The combined use case modifies the model in place and leaves it structurally pruned.** Load a fresh model for each independent comparison, including after a failed run that has already applied structural pruning.

## Coordinate alignment

SP preserves tensor width and sets selected channels to zero. TP physically removes channels. TransformerLens (TL) may also convert a tensor before passing it to a hook, for example from `[B, S, H*D]` to `[B, S, H, D]`, or reorder its axes.

Corresponding hook points can therefore have different shapes. Matching shapes do not guarantee matching element order: swapping two equally sized axes preserves the shape.

Each trace is independently restored to the original coordinate order. Known layout conversions are reversed first, then zeros are inserted at the positions of removed TP channels. This restores the coordinate space, not the values of the original unpruned model.

### Defects and restoration

A `Defect` describes a tensor transformation and stores the metadata needed to reverse it. Here, “defect” means a change in representation or coordinates, not a model error.

| Type | Restores |
| --- | --- |
| `RemovedChannelsDefect` | The original width of the last axis by inserting zeros at removed channel indices |
| `ReshapeDefect` | The shape before a reshape |
| `TransposeDefect` | The order of two swapped axes |
| `RearrangeDefect` | The layout using an inverse einops pattern and saved axis lengths |

`ActivationCorruption` stores defects in application order and restores the tensor in reverse order:

```text
Defects:     [RemovedChannelsDefect, ReshapeDefect]
Restoration: cached [B,S,H,D] → compact [B,S,H*D] → full [B,S,original_width]
```

The order matters: pruning indices refer to the original linear boundary, not the last axis of a tensor split into heads. Channels cannot be inserted directly into the converted layout.

Indices and supported conversion parameters are saved in the trace. Restoration must not depend on later changes to module dimensions during structural pruning.

## Package architecture

| Component | Responsibility |
| --- | --- |
| `BaseTracer` | Captures activations, gradients, and loss through TL; records conversion defects; identifies supported linear boundaries in groups |
| `TransferPruningTracer` | Captures SP traces with layout metadata and adjusts gradients for the mask |
| `TorchPruningTracer` | Captures TP traces with removed channel indices and unresolved pruning mappings |
| `PruningTrace` | Stores captured data, defects by hook name, and `unresolved_pruning_hooks` |
| `TransformerLensConversionRegistry` | Maps TL conversions to reversible layout defects |
| `GreatComparator` | Restores both traces and compares them numerically |
| `TensorComparison`, `CompairingTrace` | Store structured comparison results |
| `CompairingTrace.print_summary()` | Displays previously computed results |
| `PruneAndCaptureUsecase` | Prepares and captures an individual SP or TP trace |
| `ComparePrunedModelsUsecase` | Runs SP capture, TP capture, and comparison with one pruning plan |

Tracers do not compute difference metrics. The registry does not run the model. The comparator does not build groups, modify the model, or infer architectural relationships between hooks. Printing a report does not repeat the comparison.

`PruningTrace` extends `GradientCache`, storing activations, optional gradients, and a scalar metric alongside restoration metadata. Its `to_cpu()` method moves detached tensors to CPU **in place** and returns the same trace.

## Mapping TP groups to hooks

Channel indices come from the actual TP dependencies after architecture-specific corrections to the pruning plan. For supported `LinearBridge` boundaries, the operation direction determines the hook:

- Input-channel pruning maps to `hook_in`.
- Output-channel pruning maps to `hook_out`.

Modules are identified by their paths in the dependency graph, rather than dependency display names or similar hook names. For LoRA, the mapping uses the active adapter’s external boundaries: the input of A and the output of B. Internal rank coordinates and other adapters are not mapped to external hooks.

Indices from a neighboring Linear are not automatically propagated to arbitrary intermediate points. Attention scores, rotary representations, and normalization outputs may use different coordinate spaces. A dependency on a normalization parameter does not provide a general restoration rule for its hooks.

When a mapping is unknown, the TP tracer preserves the tensor and known conversion defects, marks the hook as unresolved, and emits a warning. The comparator returns `UNRESOLVED`; matching shapes alone do not justify `OK`.

### Hook selection and pruning history

Tensor capture and pruning-coordinate restoration are separate operations:

- With `names_filter=None`, the use case selects affected linear boundaries from the groups, including dependencies, rather than every model hook.
- An explicit filter can capture additional points without making their pruning mappings known.
- Given the complete list of groups, a supported canonical linear boundary without a pruning operation is treated as unchanged. It does not have to be unresolved.
- Intermediate points outside the supported mapping remain unresolved even when their layout conversion is known.

Without group history, removed channels generally cannot be restored from an already pruned model. The public `BaseTracer` path with `groups=None` currently returns conversion metadata only and an empty unresolved set. This is an implementation limitation, not a guarantee that no pruning occurred. Pass the actual groups when comparing a structurally pruned model.

Repeated pruning of the same linear boundary requires composing indices between the old and new coordinate systems. This is not implemented; repeated boundaries in the supplied groups are rejected.

## TransformerLens conversions

TL may pass a hook callback a tensor in a canonical layout that differs from the module’s native layout. `TransformerLensConversionRegistry` describes the inverse mapping through defects.

Rules are selected by the conversion class’s `(module, class name)`. This supports classes defined inside TL methods without tying semantics to substrings such as `q_proj` in hook names.

Supported conversions include:

- `ChainTensorConversion`, including nested chains;
- reversible `RearrangeTensorConversion` and 2D transpose;
- supported projection-tensor conversions into heads;
- rotary transpose;
- joint QKV and GPT BigCode MQA variants;
- OPT token unflattening;
- unambiguous `AttentionAutoConversion` cases.

Chains are analyzed backward from the captured shape, restoring intermediate shapes. The resulting defects are stored in forward application order. Inverse rearrangements and intermediate shapes are checked with `meta` tensors, without copying cached activations or allocating storage for their contents.

A conversion may have performed no operation. For example, splitting into heads may be skipped when the expected width no longer matches after TP. A TP tensor cannot automatically be assumed to use the native layout, nor can every hook be assumed to have converted its tensor.

Unknown conversions, input/output filters, and ambiguous 4D `AttentionAutoConversion` cases raise `UnsupportedHookConversionError` with the hook name. Each new conversion requires an explicit inverse rule; shape agreement is insufficient.

Supported layout transformations are bijective, so the same inverse mapping applies to activations and gradients. Arbitrary numerical conversions may instead require a separate gradient transformation through the Jacobian.

The registry relies on bridge contracts and captured shapes, rather than observing every conversion execution. For example, projection-to-heads conversion assumes a native `[B,S,H*D]` tensor. If the native tensor already had explicit head axes, the cached tensor alone may not distinguish these cases.

## Gradient capture

### Upstream anchors for converted hooks

TL capture has a reproduced limitation: requesting only converted hook points can return `None` for gradients.

A callback conversion may create a view that the subsequent model forward pass does not use:

```text
native tensor ───────────────────────→ downstream → loss
       └→ conversion → captured view
```

Calling `retain_grad()` on this side view does not resolve the issue. Actual gradients are obtained from backward hooks, which also convert the gradient layout.

TL invokes backward through `torch.autograd.grad(metric, captured_tensors, allow_unused=True)`. If the requested tensors do not include a suitable point on the actual path to the loss, backward does not traverse the required graph regions and the hooks do not fire. This call neither populates nor overwrites parameter `.grad` fields, but it requires suitable captured inputs.

With `compute_gradient=True`, include a real upstream tensor with `requires_grad=True` on the path to the loss for each required branch. One example is canonical `attn.hook_in` when its 3D `[B,S,hidden]` tensor remains on the actual computation path. A conversion does not have to be `None`; connectivity to the forward graph is what matters. A frozen input or ambiguous 4D conversion does not provide that guarantee.

Capturing the final output does not force backward to traverse every earlier point. Automatic hook selection from groups does not add an upstream anchor. Activation-only capture does not require this workaround.

The limitation is covered by `test_pruning_conversion_backward_requires_upstream_anchor`, with an independent autograd check. A missing gradient receives `NO_GRAD` and is not treated as a numerical zero or a successful comparison.

### SP masks and gradient coordinates

For `y = x * mask`, the gradient with respect to `y` may be nonzero at removed coordinates. The restored TP gradient should be compared with the gradient with respect to `x`:

```text
dL/dx = dL/dy * mask
```

`TransferPruningTracer` adjusts SP gradients using pruning indices. Comparison covers **all restored coordinates**, including removed ones. There are no `comparison_scope` / `gradient_scope` modes, retained-only masking, or excluded-element counters; differences at removed coordinates contribute to the metrics.

The current correction needs further validation: `gradient[idxs] = 0` indexes the first axis of the cached tensor, whereas group indices refer to channels in the native layout, usually on the last axis. Head conversions also require accounting for the layout transformation. The intended gradient normalization does not imply that the current indexing is correct for every tensor shape.

## Comparison results

`GreatComparator.compare_traces()` matches entries by exact hook name and restores each side using its own defects. The comparator contains no architectural heuristics for aligning neighboring points.

Calculations run on CPU after detaching tensors. FP16/BF16 values are promoted to FP32 before subtraction; FP64 is preserved when present. This avoids additional BF16 rounding of the difference. Norms and squared differences are computed in FP64 to reduce the risk of overflow.

Each point records original and restored shapes, element count, `max_abs`, `mean_abs`, `rmse`, and absolute and relative L2 norms. `OK` / `DIFF` are determined with `torch.allclose`, using the configured `atol` and `rtol` relative to the reference.

| Status | Meaning |
| --- | --- |
| `OK` | The restored tensors match within the configured tolerances |
| `DIFF` | The restored tensors differ beyond the configured tolerances |
| `MISSING` | The point is absent from one trace |
| `NO_GRAD` | At least one trace has no gradient |
| `UNRESOLVED` | The pruning or layout mapping is unknown |
| `SHAPE` | Shapes still differ after restoration |
| `RESTORE_ERROR` | Applying the defects failed |
| `NONFINITE` | The data or difference contains NaN/Inf |
| `EMPTY` | There are no elements to compare |

`CompairingTrace` supports programmatic analysis without printing. `print_summary()` separately displays FWD/BWD results, metrics, and the first `DIFF` entry. Incomplete comparisons are counted separately. “First” refers to trace entry order, not a proven causal order of errors.

Correct alignment does not guarantee bitwise equality between SP and TP. A full-width GEMM with zeroed channels and a compact GEMM can round differently in FP16/BF16. Comparison therefore records difference magnitudes and uses explicit numerical tolerances.

## Model lifecycle

`ComparePrunedModelsUsecase` performs the following steps:

1. Switches the model to evaluation mode and resets nonpermanent hooks.
2. Builds groups and structural setups once. SP and TP use the same indices, including for pruning tasks specified as fractions.
3. Installs SP masks with `rescale=False`, captures the trace, and moves it to CPU.
4. Calls `model.reset_hooks()` in `finally`, including when SP capture fails.
5. Applies structural setups and calls `group.prune()`.
6. Restores the saved CPU/CUDA RNG states for TP capture, captures the TP trace, and compares the results.
7. Restores the original module training flags.

Hook cleanup uses the standard reset behavior, as in `compare_utils`. User-defined nonpermanent hooks are also removed.

The model is not copied and remains structurally pruned after the call. Structural setups and pruning are not rolled back on error. A new model is required for each independent comparison, even if the previous run failed during comparison. `reset_hooks()` does not restore weights or channel dimensions.

`torch.inference_mode()` is unsupported: dependency graph construction requires autograd even when gradient comparison is disabled.

## DeepSeek example

The notebook `experiments/compare_tp_and_our/test_deepseek_pruning_usecase.ipynb` demonstrates loading DeepSeek-V2-Lite-Chat, preparing input tokens, and comparing the first, middle, and last layers with the combined use case. It requires neither an additional dataset nor the older `full_attention_test_with_prune` helper.

`compare_transfer_and_torch_prunings_on_layers` selects all hooks by layer prefix. It does not automatically enable disabled TL hooks: selecting `hook_mlp_in` with `use_hook_mlp_in=False`, for example, raises an error. For an attention experiment, use the main use case with a `blocks.<layer>.attn.` filter. Broader capture requires explicitly enabling the necessary hooks.

A broad filter does not provide known pruning mappings for every intermediate point; some results may be `UNRESOLVED`.

Repeated DeepSeek pruning on the same model has produced `qk_nope_head_dim=126` with `v_head_dim=128`: Q/K were already pruned while V retained its previous width. The exception’s suggestion to use `kv_lora_idxs` does not resolve this case in the current builder, because the check runs before that parameter is processed. Reload the model before an independent run.

## Tests and known limitations

The [`tests/pruning_lens`](../tests/pruning_lens) suite covers shape and coordinate restoration, conversion chains and metadata snapshots, ambiguous conversions, linear hooks and LoRA boundaries, the upstream-gradient limitation, metrics and statuses, and use cases.

The original documentation was prepared without running the complete test suite or DeepSeek inference. Test coverage alone does not establish that all tests pass against the latest API.

Areas requiring validation include:

- Synchronizing tests with the current API.
- Checking channel indexing in SP gradient correction, including converted head layouts.
- Reconciling `groups=None` behavior with the contract for unknown pruning history.

Other limitations include unsupported repeated pruning of the same linear boundary, unresolved mappings for unsupported intermediate hooks, and the need for upstream anchors when capturing gradients through converted hooks.
