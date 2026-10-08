"""Tests for pruning_lens.adapter.torch_pruning.torch_pruning_tracer."""

from types import SimpleNamespace
import warnings

import pytest
import torch
import torch.nn as nn
import torch_pruning as tp
from peft import LoraConfig
from peft.tuners.lora.layer import Linear as LoraLinear
from transformer_lens.HookedRootModule import HookedRootModule
from transformer_lens.hook_points import HookPoint
from transformer_lens.conversion_utils.conversion_steps.chain_tensor_conversion import ChainTensorConversion
from transformer_lens.conversion_utils.conversion_steps.rearrange_tensor_conversion import RearrangeTensorConversion
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge
from transformer_lens.model_bridge.generalized_components.base import GeneralizedComponent
from transformer_lens.tools.analysis.attribution_patching import GradientCache

from pruning_lens.adapter.torch_pruning.torch_pruning_tracer import (
    TorchPruningTracer,
    UnknownPruningMappingWarning,
)
from pruning_lens.core.activation_corruption import RemovedChannelsDefect


class LinearModel(HookedRootModule):
    def __init__(self, lora=False):
        super().__init__()
        # Like an embedding/upstream layer: shared linear inputs belong to autograd.
        self.stem = nn.Linear(6, 6, bias=False)
        original = nn.Linear(6, 6, bias=False)
        if lora:
            original = LoraLinear(
                original, "default", LoraConfig(r=2, lora_alpha=2), r=2, lora_alpha=2
            )
        self.first = LinearBridge(name="first")
        self.first.set_original_component(original)
        self.last = LinearBridge(name="last")
        self.last.set_original_component(nn.Linear(6, 3, bias=False))
        self.hook_intermediate = HookPoint()
        self.setup()

    def forward(self, x):
        return self.last(self.hook_intermediate(self.first(self.stem(x))))

    def loss_fn(self, logits, tokens):
        return logits.square().sum()


def cache_for(**activations):
    return GradientCache(activations, {name: None for name in activations}, torch.tensor(0.0))


def group_for(model, *operations):
    '''Synthetic groups for metadata/error tests; integration tests build real DGs.'''
    group = tp.Group()
    group._DG = tp.DependencyGraph()
    group._DG._module2name = {module: name for name, module in model.named_modules()}
    for module, handler, idxs in operations:
        group.add_dep(SimpleNamespace(target=SimpleNamespace(module=module), handler=handler), idxs)
    return group


def make_real_group(model, tokens, module, handler, idxs):
    DG = tp.DependencyGraph().build_dependency(model, example_inputs=tokens)
    return DG.get_pruning_group(module, handler, idxs=idxs)


@pytest.mark.parametrize("compute_gradient", [False, True])
def test_real_pruning_capture_and_restore(compute_gradient):
    model = LinearModel()
    tokens = torch.randn(2, 3, 6)
    original_first_out = model.first(model.stem(tokens)).detach()
    group = make_real_group(
        model, tokens, model.first.original_component, tp.prune_linear_out_channels, [1, 4]
    )
    group.prune()
    with pytest.warns(UnknownPruningMappingWarning, match="hook_intermediate") as caught:
        trace = TorchPruningTracer.capture_hookpoints_activations_with_defects(
            model, tokens, names_filter=lambda _: True,
            compute_gradient=compute_gradient, groups=[group],
        )
    assert len(caught) == 1
    assert trace.unresolved_pruning_hooks == {"hook_intermediate"}
    assert trace.corruptions["first.hook_in"].defects == []
    assert trace.corruptions["last.hook_out"].defects == []
    expected = original_first_out.clone()
    expected[..., [1, 4]] = 0
    for name in ("first.hook_out", "last.hook_in"):
        corruption = trace.corruptions[name]
        assert len(corruption.defects) == 1
        assert corruption.defects[0].restore_info.tolist() == [1, 4]
        restored = corruption.restore_activation(trace.activations[name])
        torch.testing.assert_close(restored, expected)
        if compute_gradient:
            grad = corruption.restore_activation(trace.gradients[name])
            assert grad.shape == expected.shape
            assert torch.count_nonzero(grad[..., [1, 4]]) == 0
        else:
            assert trace.gradients[name] is None
    assert "hook_intermediate" in trace.activations


def test_conversion_is_undone_before_inserting_channels():
    model = LinearModel()
    tokens = torch.randn(2, 3, 6)
    group = make_real_group(
        model, tokens, model.first.original_component, tp.prune_linear_out_channels, [1, 4]
    )
    group.prune()
    model.first.hook_out.hook_conversion = RearrangeTensorConversion("b s (h d) -> b s h d", h=2)
    native = model.first.original_component(model.stem(tokens)).detach()
    trace = TorchPruningTracer.capture_hookpoints_activations_with_defects(
        model, tokens, names_filter=["first.hook_out"], compute_gradient=False, groups=[group]
    )
    assert trace.activations["first.hook_out"].shape == (2, 3, 2, 2)
    corruption = trace.corruptions["first.hook_out"]
    assert isinstance(corruption.defects[0], RemovedChannelsDefect)
    restored = corruption.restore_activation(trace.activations["first.hook_out"])
    assert restored.shape == (2, 3, 6)
    assert torch.equal(restored[..., [0, 2, 3, 5]], native)


@pytest.mark.parametrize("hook_name", ["first.hook_out", "last.hook_in"])
@pytest.mark.parametrize("transpose_heads", [False, True])
@pytest.mark.parametrize("capture_upstream", [False, True])
def test_pruning_conversion_backward_requires_upstream_anchor(
    hook_name, transpose_heads, capture_upstream,
):
    torch.manual_seed(42)
    model = LinearModel()
    tokens = torch.randn(2, 3, 6)
    removed_idxs = [1, 4]
    kept_idxs = [0, 2, 3, 5]

    # Save full-width data before TP changes either adjacent Linear.
    full_native = model.first.original_component(model.stem(tokens)).detach().requires_grad_()
    full_last_weight = model.last.original_component.weight.detach().clone()
    group = make_real_group(
        model, tokens, model.first.original_component,
        tp.prune_linear_out_channels, removed_idxs,
    )
    group.prune()

    conversion = RearrangeTensorConversion("b s (h d) -> b s h d", h=2)
    if transpose_heads:
        conversion = ChainTensorConversion([
            conversion,
            RearrangeTensorConversion("b s h d -> b h s d"),
        ])
    model.hook_dict[hook_name].hook_conversion = conversion

    # Independent autograd reference: no HookPoints or conversions involved.
    native = model.first.original_component(model.stem(tokens)).detach().requires_grad_()
    native_logits = torch.nn.functional.linear(native, model.last.original_component.weight)
    native_loss = model.loss_fn(native_logits, tokens)
    native_grad, = torch.autograd.grad(native_loss, native)

    # Full-width masked reference. Differentiate w.r.t. the *pre-mask* tensor,
    # so removed coordinates have zero gradients, as in restored TP tensors.
    mask = torch.ones(6)
    mask[removed_idxs] = 0
    masked_logits = torch.nn.functional.linear(full_native * mask, full_last_weight)
    masked_loss = model.loss_fn(masked_logits, tokens)
    masked_grad, = torch.autograd.grad(masked_loss, full_native)

    # Current TL capture needs a real upstream tensor to drive backward through
    # converted points. Without it, None is the documented capture limitation.
    names = ["first.hook_in", hook_name] if capture_upstream else [hook_name]
    trace = TorchPruningTracer.capture_hookpoints_activations_with_defects(
        model, tokens, names_filter=names, compute_gradient=True, groups=[group],
    )
    assert not trace.unresolved_pruning_hooks
    torch.testing.assert_close(trace.metric, native_loss.detach())
    torch.testing.assert_close(trace.metric, masked_loss.detach())
    torch.testing.assert_close(trace.activations[hook_name], conversion.convert(native.detach()))
    corruption = trace.corruptions[hook_name]
    assert isinstance(corruption.defects[0], RemovedChannelsDefect)
    assert len(corruption.defects) == (3 if transpose_heads else 2)
    restored_activation = corruption.restore_activation(trace.activations[hook_name])
    assert restored_activation.shape == (2, 3, 6)
    torch.testing.assert_close(restored_activation[..., kept_idxs], native.detach())
    assert torch.count_nonzero(restored_activation[..., removed_idxs]) == 0
    torch.testing.assert_close(restored_activation, full_native.detach() * mask)

    captured_grad = trace.gradients[hook_name]
    if not capture_upstream:
        assert captured_grad is None
        return

    assert captured_grad is not None
    torch.testing.assert_close(captured_grad, conversion.convert(native_grad))
    restored_grad = corruption.restore_activation(captured_grad)
    assert restored_grad.shape == (2, 3, 6)
    torch.testing.assert_close(restored_grad[..., kept_idxs], native_grad)
    assert torch.count_nonzero(restored_grad[..., removed_idxs]) == 0
    torch.testing.assert_close(restored_grad, masked_grad)


def test_canonical_hook_indices_are_snapshotted():
    model = LinearModel()
    module = model.first.original_component
    group = group_for(model, (module, tp.prune_linear_out_channels, [7, 2, 7]))
    activation = torch.randn(1, 2, 6)
    cache = cache_for(**{"first.hook_out": activation})
    trace = TorchPruningTracer._append_defects(cache, model, [group])
    group[0].idxs.clear()
    assert trace.activations is cache.activations
    assert trace.gradients is cache.gradients
    assert trace.metric is cache.metric
    assert trace.unresolved_pruning_hooks == set()
    for name in cache.activations:
        assert trace.corruptions[name].defects[0].restore_info.tolist() == [2, 7]


def test_unknown_groups_are_distinct_from_known_empty_groups():
    model = LinearModel()
    cache = cache_for(**{"first.hook_out": torch.randn(1, 2, 6)})
    with pytest.warns(UnknownPruningMappingWarning, match="not provided"):
        unknown = TorchPruningTracer._append_defects(cache, model, groups=None)
    with warnings.catch_warnings(record=True) as caught:
        known = TorchPruningTracer._append_defects(cache, model, groups=[])
    assert not caught
    assert unknown.unresolved_pruning_hooks == {"first.hook_out"}
    assert known.unresolved_pruning_hooks == set()
    assert known.corruptions["first.hook_out"].defects == []


def test_group_direction_selects_the_correct_hook_of_the_same_linear():
    model = LinearModel()
    module = model.first.original_component
    group = group_for(model,
        (module, tp.prune_linear_in_channels, [0]),
        (module, tp.prune_linear_out_channels, [4]),
    )
    cache = cache_for(**{
        "first.hook_in": torch.randn(1, 2, 6),
        "first.hook_out": torch.randn(1, 2, 6),
    })
    trace = TorchPruningTracer._append_defects(cache, model, [group])
    assert trace.corruptions["first.hook_in"].defects[0].restore_info.tolist() == [0]
    assert trace.corruptions["first.hook_out"].defects[0].restore_info.tolist() == [4]


def test_alias_is_unresolved_but_canonical_hook_receives_defect():
    model = LinearModel()
    model.hook_dict["hook_q"] = model.first.hook_out
    group = group_for(model, (model.first.original_component, tp.prune_linear_out_channels, [1]))
    names = ["first.hook_out", "hook_q"]
    cache = cache_for(**{name: torch.randn(1, 2, 6) for name in names})
    with pytest.warns(UnknownPruningMappingWarning, match="hook_q"):
        trace = TorchPruningTracer._append_defects(cache, model, [group])
    assert set(trace.corruptions) == set(names)
    assert trace.unresolved_pruning_hooks == {"hook_q"}
    assert trace.corruptions["hook_q"].defects == []
    assert trace.corruptions["first.hook_out"].defects[0].restore_info.tolist() == [1]


def test_dependency_display_name_is_not_used_as_module_path():
    model = LinearModel()
    group = group_for(model, (model.first.original_component, tp.prune_linear_out_channels, [1]))
    group[0].dep.target.name = "not.a.module.path (Linear(in_features=6, out_features=6))"
    cache = cache_for(**{"first.hook_out": torch.randn(1, 2, 6)})
    trace = TorchPruningTracer._append_defects(cache, model, [group])
    assert trace.corruptions["first.hook_out"].defects[0].restore_info.tolist() == [1]


def test_intermediate_hook_keeps_conversions_but_not_neighbor_indices():
    model = LinearModel()
    model.hook_intermediate.hook_conversion = RearrangeTensorConversion("b s (h d) -> b s h d", h=2)
    cache = cache_for(hook_intermediate=torch.randn(1, 3, 2, 3))
    group = group_for(model, (model.first.original_component, tp.prune_linear_out_channels, [2]))
    with pytest.warns(UnknownPruningMappingWarning):
        trace = TorchPruningTracer._append_defects(cache, model, [group])
    assert trace.unresolved_pruning_hooks == {"hook_intermediate"}
    assert len(trace.corruptions["hook_intermediate"].defects) == 1
    assert not isinstance(trace.corruptions["hook_intermediate"].defects[0], RemovedChannelsDefect)


def test_norm_parameter_in_group_does_not_imply_linear_coverage():
    model = LinearModel()
    model.norm = GeneralizedComponent(name="norm")
    model.norm.set_original_component(nn.LayerNorm(6))
    model.setup()
    group = group_for(model, (model.norm.original_component.weight, tp.prune_parameter_out_channels, [1]))
    cache = cache_for(**{
        "norm.hook_in": torch.randn(1, 2, 6),
        "norm.hook_out": torch.randn(1, 2, 6),
    })
    with pytest.warns(UnknownPruningMappingWarning) as caught:
        trace = TorchPruningTracer._append_defects(cache, model, [group])
    assert len(caught) == 1
    assert trace.unresolved_pruning_hooks == set(cache.activations)
    assert all(not corruption.defects for corruption in trace.corruptions.values())


@pytest.mark.parametrize("direction", ["in", "out", "rank"])
def test_lora_real_groups_respect_outer_and_rank_dimensions(direction):
    model = LinearModel(lora=True)
    tokens = torch.randn(2, 3, 6)
    lora = model.first.original_component
    if direction == "rank":
        module, handler, idxs = lora.lora_A["default"], tp.prune_linear_out_channels, [1]
    else:
        module = lora.base_layer
        handler = tp.prune_linear_in_channels if direction == "in" else tp.prune_linear_out_channels
        idxs = [1, 4]
    group = make_real_group(model, tokens, module, handler, idxs)
    group.prune()
    trace = TorchPruningTracer.capture_hookpoints_activations_with_defects(
        model, tokens, names_filter=["first.hook_in", "first.hook_out"],
        compute_gradient=False, groups=[group],
    )
    for boundary in ("in", "out"):
        defects = trace.corruptions[f"first.hook_{boundary}"].defects
        if direction == boundary:
            assert defects[0].restore_info.tolist() == idxs
        else:
            assert defects == []


@pytest.mark.parametrize("direction", ["in", "out"])
def test_lora_uses_selected_adapter_boundary_without_checking_base(direction):
    model = LinearModel(lora=True)
    lora = model.first.original_component
    handler = tp.prune_linear_in_channels if direction == "in" else tp.prune_linear_out_channels
    adapter = lora.lora_A["default"] if direction == "in" else lora.lora_B["default"]
    group = group_for(model, (adapter, handler, [1]), (lora.base_layer, handler, [2]))
    cache = cache_for(**{f"first.hook_{direction}": torch.randn(1, 2, 6)})
    trace = TorchPruningTracer._append_defects(cache, model, [group])
    assert trace.corruptions[f"first.hook_{direction}"].defects[0].restore_info.tolist() == [1]


@pytest.mark.parametrize("active_adapter", ["default", "second"])
def test_lora_uses_only_the_current_active_adapter(active_adapter):
    model = LinearModel(lora=True)
    lora = model.first.original_component
    lora.update_layer("second", r=2, lora_alpha=2, config=LoraConfig(r=2, lora_alpha=2))
    lora.set_adapter(active_adapter)
    group = group_for(model,
        (lora.lora_B["default"], tp.prune_linear_out_channels, [1]),
        (lora.lora_B["second"], tp.prune_linear_out_channels, [2]),
    )
    cache = cache_for(**{"first.hook_out": torch.randn(1, 2, 6)})
    trace = TorchPruningTracer._append_defects(cache, model, [group])
    expected = [1] if active_adapter == "default" else [2]
    assert trace.corruptions["first.hook_out"].defects[0].restore_info.tolist() == expected


@pytest.mark.parametrize("second_idxs", [[1], [2]])
def test_repeated_boundary_pruning_is_rejected(second_idxs):
    model = LinearModel()
    module = model.first.original_component
    groups = [group_for(model, (module, tp.prune_linear_out_channels, idxs)) for idxs in ([1], second_idxs)]
    cache = cache_for(**{"first.hook_out": torch.randn(1, 2, 6)})
    with pytest.raises(ValueError, match="Repeated channel pruning"):
        TorchPruningTracer._append_defects(cache, model, groups)


@pytest.mark.parametrize("idxs", [[-1], [99]])
def test_invalid_indices_are_rejected_when_restoring(idxs):
    model = LinearModel()
    group = group_for(model, (model.first.original_component, tp.prune_linear_out_channels, idxs))
    cache = cache_for(**{"first.hook_out": torch.randn(1, 2, 6)})
    trace = TorchPruningTracer._append_defects(cache, model, [group])
    with pytest.raises(ValueError, match="outside"):
        trace.corruptions["first.hook_out"].restore_activation(cache.activations["first.hook_out"])


def test_indices_for_another_module_are_not_assigned_to_the_hook():
    model = LinearModel()
    group = group_for(model, (nn.Linear(6, 6), tp.prune_linear_out_channels, [1]))
    cache = cache_for(**{"first.hook_out": torch.randn(1, 2, 6)})
    trace = TorchPruningTracer._append_defects(cache, model, [group])
    assert trace.corruptions["first.hook_out"].defects == []


def test_capture_shape_is_not_revalidated_against_model_weights():
    model = LinearModel()
    cache = cache_for(**{"first.hook_out": torch.randn(1, 2, 5)})
    trace = TorchPruningTracer._append_defects(cache, model, [])
    assert trace.activations is cache.activations
    assert trace.corruptions["first.hook_out"].defects == []
