from types import SimpleNamespace

import pytest
import torch
import torch_pruning as tp
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge

from lora_transfer_pruning.adapter.torch_pruning.universal.universal_group_builder import UniversalGroupBuilder


@pytest.mark.parametrize('prune_out', [False, True])
def test_nonempty_indices_still_require_matching_boundary(prune_out):
    bridge = LinearBridge(name='projection')
    bridge.set_original_component(torch.nn.Linear(4, 4))
    group = tp.Group()
    # An operation on the same module in the other direction is insufficient.
    other_handler = tp.prune_linear_in_channels if prune_out else tp.prune_linear_out_channels
    group.add_dep(SimpleNamespace(target=SimpleNamespace(module=bridge.original_component),
                                  handler=other_handler), [2])
    with pytest.raises(AssertionError, match='Expected 1 group items'):
        UniversalGroupBuilder.replace_linear_indices(group, bridge, [1], prune_out, tp.DependencyGraph())
    assert group[0].idxs == [2]
