import pytest
import torch
from transformer_lens.conversion_utils.conversion_steps.rearrange_tensor_conversion import RearrangeTensorConversion

from pruning_lens.adapter.base_tracer import BaseTracer


@pytest.mark.parametrize("converted", [False, True])
def test_local_sum_gradient_and_hook_cleanup(tiny_pruning_model, converted):
    model = tiny_pruning_model()
    if converted:
        for point in (model.first.hook_in, model.first.hook_out):
            point.hook_conversion = RearrangeTensorConversion("b s (h d) -> b s h d", h=2)
    model.first.hook_out.add_hook(lambda tensor, hook: tensor * 2)
    handles = list(model.first.hook_out.fwd_hooks)
    tokens = torch.randn(2, 3, 6)
    with torch.no_grad():
        cache = BaseTracer._cache_local_activation_and_gradient(
            model, tokens, ["first.hook_out", "last.hook_in", "first.hook_in", "last.hook_out"],
        )
    x = model.stem(tokens).detach().requires_grad_(True)
    y = model.first.original_component(x) * 2
    expected, = torch.autograd.grad(y.sum(), x)
    assert cache.gradients["first.hook_in"].shape == cache.activations["first.hook_in"].shape
    torch.testing.assert_close(cache.gradients["first.hook_in"].reshape_as(x), expected)
    torch.testing.assert_close(cache.activations["first.hook_out"].reshape_as(y), y.detach())
    torch.testing.assert_close(cache.gradients["first.hook_out"], torch.ones_like(cache.activations["first.hook_out"]))
    expected_metric = sum(value.double().sum() for name, value in cache.activations.items() if name.endswith("hook_out"))
    torch.testing.assert_close(cache.metric, expected_metric)
    assert model.first.hook_out.fwd_hooks == handles
    assert not model.first.hook_in.fwd_hooks
    assert all(p.grad is None for p in model.parameters())
    assert all(not value.requires_grad for value in [*cache.activations.values(), *cache.gradients.values(), cache.metric] if value is not None)


def test_local_gradient_ignores_downstream_weights(tiny_pruning_model):
    model = tiny_pruning_model()
    tokens = torch.randn(1, 3, 6)
    first = BaseTracer._cache_local_activation_and_gradient(model, tokens, ["first.hook_in", "first.hook_out"])
    with torch.no_grad():
        model.last.original_component.weight.mul_(100)
    second = BaseTracer._cache_local_activation_and_gradient(model, tokens, ["first.hook_in", "first.hook_out"])
    torch.testing.assert_close(first.gradients["first.hook_in"], second.gradients["first.hook_in"])


def test_repeated_calls_rejected(tiny_pruning_model):
    model = tiny_pruning_model()
    model.forward = lambda tokens: model.first(model.first(tokens))
    with pytest.raises(ValueError, match="repeated calls"):
        BaseTracer._cache_local_activation_and_gradient(model, torch.randn(1, 3, 6), ["first.hook_in", "first.hook_out"])
    assert not model.first.hook_in.fwd_hooks
    assert not model.first.hook_out.fwd_hooks


def test_frozen_input(tiny_pruning_model):
    model = tiny_pruning_model()
    cache = BaseTracer._cache_local_activation_and_gradient(model, torch.randn(1, 3, 6), ["stem.hook_in", "stem.hook_out"])
    assert cache.gradients["stem.hook_in"] is None
    assert cache.gradients["stem.hook_out"] is not None


def test_input_mask_boundary_is_post_mask(tiny_pruning_model):
    model = tiny_pruning_model()
    model.first.hook_in.add_hook(lambda tensor, hook: tensor * 0)
    cache = BaseTracer._cache_local_activation_and_gradient(model, torch.randn(1, 3, 6), ["first.hook_in", "first.hook_out"])
    assert torch.count_nonzero(cache.activations["first.hook_in"]) == 0
    expected = model.first.original_component.weight.sum(0).expand_as(cache.activations["first.hook_in"])
    torch.testing.assert_close(cache.gradients["first.hook_in"], expected)


def test_failure_cleanup_and_validation(tiny_pruning_model):
    model = tiny_pruning_model()
    with pytest.raises(ValueError, match="missing hook_out"):
        BaseTracer._cache_local_activation_and_gradient(model, torch.randn(1, 3, 6), ["first.hook_in", "first.hook_in"])
    with pytest.raises(RuntimeError, match="did not both execute"):
        model.forward = lambda tokens: tokens
        BaseTracer._cache_local_activation_and_gradient(model, torch.randn(1, 3, 6), ["first.hook_in", "first.hook_out"])
    assert not model.first.hook_in.fwd_hooks
    assert not model.first.hook_out.fwd_hooks


@pytest.mark.parametrize("names, message", [
    (["first"], "Unknown registered hook"),
    (["first.hook_in"], "missing hook_out"),
    (["first.hook_out"], "missing hook_in"),
])
def test_requires_explicit_hook_pairs(tiny_pruning_model, names, message):
    model = tiny_pruning_model()
    with pytest.raises(ValueError, match=message):
        BaseTracer._cache_local_activation_and_gradient(model, torch.randn(1, 3, 6), names)
    assert not model.first.hook_in.fwd_hooks
    assert not model.first.hook_out.fwd_hooks


def test_hook_aliases_use_canonical_module_name(tiny_pruning_model):
    model = tiny_pruning_model()
    model.hook_dict["alias_input"] = model.first.hook_in
    model.hook_dict["alias_output"] = model.first.hook_out
    cache = BaseTracer._cache_local_activation_and_gradient(
        model, torch.randn(1, 3, 6), ["alias_input", "alias_output", "first.hook_in", "first.hook_out", "alias_input"],
    )
    assert set(cache.activations) == {"first.hook_in", "first.hook_out"}
    assert set(cache.gradients) == set(cache.activations)
    assert cache.gradients["first.hook_in"] is not None
