"""Tests for pruning_lens.adapter.transfer_pruning.transfer_pruning_tracer."""

from types import SimpleNamespace

import pytest
import torch
from transformer_lens.conversion_utils.conversion_steps.base_tensor_conversion import BaseTensorConversion
from transformer_lens.HookedRootModule import HookedRootModule
from transformer_lens.hook_points import HookPoint
from transformer_lens.tools.analysis.attribution_patching import GradientCache

from pruning_lens.adapter.transfer_pruning.transfer_pruning_tracer import TransferPruningTracer
from pruning_lens.core.transformer_lens_conversion_registry import UnsupportedHookConversionError


def test_append_defects_preserves_cache_and_only_visits_captured_hooks(attention_block_with_conversions):
    head = attention_block_with_conversions.q.hook_out
    model = SimpleNamespace(hook_dict={"q": head, "identity": HookPoint(), "unused": HookPoint()})
    source = torch.randn(2, 5, 6, dtype=torch.bfloat16)
    cache = GradientCache(
        activations={"q": head.hook_conversion.convert(source), "identity": source},
        gradients={"q": head.hook_conversion.convert(source), "identity": None},
        metric=torch.tensor(1.0),
    )
    trace = TransferPruningTracer._append_defects(cache, model)
    assert trace.activations is cache.activations
    assert trace.gradients is cache.gradients
    assert trace.metric is cache.metric
    assert set(trace.corruptions) == {"q", "identity"}
    assert trace.corruptions["identity"].defects == []
    for tensor in (trace.activations["q"], trace.gradients["q"]):
        restored = trace.corruptions["q"].restore_activation(tensor)
        assert torch.equal(restored, source)
        assert restored.dtype == source.dtype


def test_error_includes_hook_name():
    hook = HookPoint()
    hook.hook_conversion = BaseTensorConversion()
    cache = GradientCache({"bad_hook": torch.ones(2, 3)}, {}, torch.tensor(0.0))
    with pytest.raises(UnsupportedHookConversionError, match="bad_hook"):
        TransferPruningTracer._append_defects(cache, SimpleNamespace(hook_dict={"bad_hook": hook}))


def test_missing_hook_raises():
    cache = GradientCache({"missing_hook": torch.ones(2, 3)}, {}, torch.tensor(0.0))
    with pytest.raises(KeyError, match="missing_hook"):
        TransferPruningTracer._append_defects(cache, SimpleNamespace(hook_dict={}))


class TinyModel(HookedRootModule):
    def __init__(self, attention_conversions):
        super().__init__()
        self.proj = torch.nn.Linear(6, 6, bias=False)
        self.hook_start = HookPoint()
        self.hook_heads = attention_conversions.q.hook_out
        self.hook_end = HookPoint()
        self.setup()

    def forward(self, tokens):
        x = self.hook_start(self.proj(tokens))
        return self.hook_end(self.hook_heads(x) * 2)

    def loss_fn(self, logits, tokens):
        return logits.square().sum()


@pytest.mark.parametrize("compute_gradient", [True, False])
def test_real_capture_forward_backward(compute_gradient, attention_block_with_conversions):
    model = TinyModel(attention_block_with_conversions)
    tokens = torch.randn(2, 5, 6)
    trace = TransferPruningTracer.capture_hookpoints_activations_with_defects(
        model, tokens, names_filter=lambda _: True,
        compute_gradient=compute_gradient,
    )
    corruption = trace.corruptions["hook_heads"]
    native = corruption.restore_activation(trace.activations["hook_heads"])
    assert torch.equal(native, model.proj(tokens))
    if compute_gradient:
        grad = corruption.restore_activation(trace.gradients["hook_heads"])
        torch.testing.assert_close(grad, native * 8)
    else:
        assert all(grad is None for grad in trace.gradients.values())
    assert not model.hook_heads.fwd_hooks
    assert not model.hook_heads.bwd_hooks
