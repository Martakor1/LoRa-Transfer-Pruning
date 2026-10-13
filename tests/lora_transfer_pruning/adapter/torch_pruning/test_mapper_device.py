"""Mask indices belong to their Linear, not to a global bridge device."""

import torch
import torch_pruning as tp
from transformer_lens.HookedRootModule import HookedRootModule
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge

from lora_transfer_pruning.usecase.torch_pruning_group_mapper import TorchPruningGroupMapper


class ModelWithNoGlobalDevice(HookedRootModule):
    def __init__(self):
        super().__init__()
        self.first = LinearBridge(name='first')
        self.first.set_original_component(torch.nn.Linear(6, 6, bias=False))
        self.setup()

    @property
    def device(self):
        raise AssertionError('A mixed model has no single activation device')

    @property
    def model(self):
        return self

    def forward(self, x):
        return self.first(x)


def test_mapper_uses_target_weight_device_without_global_device():
    model = ModelWithNoGlobalDevice()
    x = torch.randn(2, 3, 6)
    expected = model(x).detach().clone()
    expected[..., [1, 4]] = 0
    graph = tp.DependencyGraph().build_dependency(model, example_inputs=x)
    group = graph.get_pruning_group(model.first._original_component,
                                   tp.prune_linear_out_channels, idxs=[1, 4])
    TorchPruningGroupMapper.prepare_group_for_transfer_pruning(model, group, rescale=False)
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
