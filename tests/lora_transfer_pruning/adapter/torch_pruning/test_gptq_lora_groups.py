"""GPTQ adapter groups drive masks on the complete projection, not only LoRA."""
from types import SimpleNamespace

import pytest
import torch
import torch_pruning as tp
from peft import LoraConfig
from peft.tuners.lora import GPTQLoraLinear
from transformer_lens.HookedRootModule import HookedRootModule
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge

from lora_transfer_pruning.adapter.pruning_propagation.torch_pruning_group_builder.universal.universal_group_builder import UniversalGroupBuilder
from lora_transfer_pruning.core.torch_pruning_group_mapper import TorchPruningGroupMapper


class BufferLinear(torch.nn.Module):
    """CPU stand-in for a frozen GPTQ base; deliberately not nn.Linear."""
    def __init__(self):
        super().__init__()
        self.in_features = self.out_features = 6
        self.register_buffer('weight', torch.randn(6, 6))
        self.bias = None

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight)


class TinyGPTQBridge(HookedRootModule):
    def __init__(self):
        super().__init__()
        self.proj = LinearBridge(name='proj')
        original = GPTQLoraLinear(BufferLinear(), 'default', LoraConfig(r=2), r=2)
        torch.nn.init.normal_(original.lora_B['default'].weight)
        self.proj.set_original_component(original)
        self.setup()

    @property
    def model(self):
        return self

    def forward(self, x):
        return self.proj(x)


@pytest.mark.parametrize('prune_out', [False, True])
def test_gptq_adapter_indices_and_outer_mask(prune_out):
    model = TinyGPTQBridge()
    original = model.proj.original_component
    target = original.lora_B['default'] if prune_out else original.lora_A['default']
    handler = tp.prune_linear_out_channels if prune_out else tp.prune_linear_in_channels
    dep = SimpleNamespace(target=SimpleNamespace(module=target,
                          name='proj._original_component.lora_B.default'),
                          layer=target, handler=handler, pruning_fn=handler)
    group = tp.Group()
    group.add_dep(dep, [0])
    UniversalGroupBuilder.replace_linear_indices(group, model.proj, [1, 4], prune_out, tp.DependencyGraph())
    assert group[0].idxs == [1, 4]
    x = torch.randn(2, 3, 6, requires_grad=True)
    if prune_out:
        expected = model(x).detach().clone()
        expected[..., [1, 4]] = 0
    else:
        masked = x.detach().clone()
        masked[..., [1, 4]] = 0
        expected = model(masked).detach()
    TorchPruningGroupMapper.prepare_group_for_transfer_pruning(model, group, rescale=False)
    actual = model(x)
    torch.testing.assert_close(actual, expected)
    point = model.proj.hook_out if prune_out else model.proj.hook_in
    assert len(point.fwd_hooks) == 1
    actual.square().sum().backward()
    grad = target.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    removed_grad = grad[[1, 4]] if prune_out else grad[:, [1, 4]]
    assert torch.count_nonzero(removed_grad) == 0
