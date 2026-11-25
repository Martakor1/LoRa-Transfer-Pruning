"""Packed bases without .weight: resolve device and mask the whole LoRA layer."""
from types import SimpleNamespace

import pytest
import torch
import torch_pruning as tp
from peft import LoraConfig
from peft.tuners.lora.layer import Linear as PeftLinear
from transformer_lens.HookedRootModule import HookedRootModule
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge

from lora_transfer_pruning.adapter.pruning_propagation.torch_pruning_group_builder.tp_utils import get_device
from lora_transfer_pruning.adapter.pruning_propagation.torch_pruning_group_builder.index_utils import IndexUtils
from lora_transfer_pruning.core.torch_pruning_group_mapper import TorchPruningGroupMapper


class PackedBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_features = self.out_features = 6
        self.register_buffer('qweight', torch.randn(6, 6))
        self.bias = None

    def forward(self, x):
        return torch.nn.functional.linear(x, self.qweight)


class PackedBridge(HookedRootModule):
    def __init__(self):
        super().__init__()
        self.proj = LinearBridge(name='proj')
        self.proj.set_original_component(PeftLinear(PackedBase(), 'default', LoraConfig(r=2, lora_alpha=4), r=2, lora_alpha=4))
        torch.nn.init.normal_(self.proj._original_component.lora_B['default'].weight)
        self.setup()

    @property
    def model(self):
        return self

    def forward(self, x):
        return self.proj(x)


@pytest.mark.parametrize('prune_out', [False, True])
def test_packed_base_device_fraction_and_outer_mask(prune_out):
    model = PackedBridge()
    original = model.proj._original_component
    assert not hasattr(original.base_layer, 'weight')
    assert get_device(model.proj) == torch.device('cpu')
    fn = tp.prune_linear_out_channels if prune_out else tp.prune_linear_in_channels
    idxs = IndexUtils.convert_linear_idx_fraction_to_idxs(0.5, model.proj, fn)
    assert len(idxs) == 3 and idxs.max() < 6
    target = original.lora_B['default'] if prune_out else original.lora_A['default']
    dep = SimpleNamespace(layer=target, pruning_fn=fn,
                          target=SimpleNamespace(name='proj._original_component.lora_B.default'))
    group = [(dep, [1, 4])]
    x = torch.randn(2, 3, 6, requires_grad=True)
    if prune_out:
        expected = model(x).detach().index_fill(-1, torch.tensor([1, 4]), 0)
    else:
        expected = model(x.detach().index_fill(-1, torch.tensor([1, 4]), 0)).detach()
    TorchPruningGroupMapper.prepare_group_for_transfer_pruning(model, group, rescale=False)
    actual = model(x)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    grad = target.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert torch.count_nonzero(grad[[1, 4]] if prune_out else grad[:, [1, 4]]) == 0
