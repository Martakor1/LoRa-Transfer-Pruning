from types import SimpleNamespace

import pytest
import torch
import torch_pruning as tp
from torch import nn
from transformer_lens.HookedRootModule import HookedRootModule

from lora_transfer_pruning.adapter.pruning_propagation.torch_pruning_group_builder.torch_pruning_group_builder import (
    TorchPruningGroupBuilder,
)
from lora_transfer_pruning.core.tl_adapters import ExpertBridge


class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = nn.Linear(4, 6, bias=False)
        self.gate_proj = nn.Linear(4, 6, bias=False)
        self.down_proj = nn.Linear(6, 4, bias=False)
        self.calls = 0
        self.input_spec = None

    def forward(self, x):
        self.calls += 1
        self.input_spec = (x.shape, x.dtype, x.requires_grad)
        return self.down_proj(self.up_proj(x) * self.gate_proj(x).sigmoid())


class SparseModel(HookedRootModule):
    def __init__(self, experts=None, route_first_expert=True):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.route_first_expert = route_first_expert
        self.experts = nn.ModuleList()
        for expert in experts if experts is not None else [Expert() for _ in range(3)]:
            bridge = ExpertBridge(name='experts')
            bridge.set_original_component(expert)
            self.experts.append(bridge)
        self.setup()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self, input, use_cache=False, return_type='logits'):
        return self.experts[0](input) if self.route_first_expert else input


def test_single_graph_contains_all_experts_and_local_dependencies():
    model = SparseModel().requires_grad_(False)
    x = torch.randn(2, 4, requires_grad=True)
    graph = TorchPruningGroupBuilder(model, x).DG

    assert [expert.calls for expert in model.experts] == [2, 1, 1]
    for expert in model.experts:
        assert expert.up_proj in graph.module2node
        assert expert.down_proj in graph.module2node
        group = graph.get_pruning_group(expert.up_proj, tp.prune_linear_out_channels, [1, 3])
        linear_items = {
            (dep.target.module, dep.handler): list(idxs)
            for dep, idxs in group if isinstance(dep.target.module, nn.Linear)
        }
        assert linear_items == {
            (expert.up_proj, tp.prune_linear_out_channels): [1, 3],
            (expert.gate_proj, tp.prune_linear_out_channels): [1, 3],
            (expert.down_proj, tp.prune_linear_in_channels): [1, 3],
        }
        assert not expert._forward_pre_hooks
        assert all(not p.requires_grad for p in expert.parameters())
    assert model.experts[1].input_spec == (torch.Size([1, 4]), torch.float32, True)


def test_normal_output_rng_and_existing_hooks_are_preserved():
    model = SparseModel().double()
    x = torch.randn(2, 4, dtype=torch.float64)
    expected = model(x).detach()
    handle = model.experts[1].register_forward_pre_hook(lambda module, args: None)
    rng = torch.get_rng_state()
    output, *extra = TorchPruningGroupBuilder._trace_forward(model, x)
    torch.testing.assert_close(output, expected)
    assert len(extra) == 3
    assert extra[0].dtype == torch.float64
    assert torch.equal(rng, torch.get_rng_state())
    assert list(model.experts[1]._forward_pre_hooks) == [handle.id]
    handle.remove()


def test_disconnected_expert_family_uses_config_input_size():
    model = SparseModel(route_first_expert=False).double().requires_grad_(False)
    x = torch.zeros(1)
    output, *extra = TorchPruningGroupBuilder._trace_forward(model, x)
    assert output is x
    assert len(extra) == 3
    assert all(expert.input_spec == (torch.Size([1, 4]), torch.float64, True)
               for expert in model.experts)


def test_expert_does_not_need_known_projection_names():
    model = SparseModel(experts=[Expert(), nn.Sequential(nn.Linear(4, 7), nn.SiLU(), nn.Linear(7, 4))])
    _, *extra = TorchPruningGroupBuilder._trace_forward(model, torch.randn(2, 4))
    assert len(extra) == 2
    assert extra[1].shape == (1, 4)


def test_experts_are_not_traced_if_forward_fails(monkeypatch):
    model = SparseModel()

    def fail(**kwargs):
        raise RuntimeError('forward failed')

    monkeypatch.setattr(model, 'forward', fail)
    with pytest.raises(RuntimeError, match='forward failed'):
        TorchPruningGroupBuilder._trace_forward(model, torch.randn(2, 4))
    assert [expert.calls for expert in model.experts] == [0, 0, 0]
    assert all(not expert._forward_pre_hooks for expert in model.experts)


def test_model_without_experts_returns_original_output():
    model = SparseModel(experts=[], route_first_expert=False)
    x = torch.ones(1, 4)
    output, = TorchPruningGroupBuilder._trace_forward(model, x)
    assert output is x


def test_builder_traces_lora_b_in_unrouted_bridged_expert():
    from peft import LoraConfig
    from peft.tuners.lora import GPTQLoraLinear
    from transformer_lens.model_bridge.generalized_components.linear import LinearBridge

    class BufferLinear(nn.Module):
        '''Frozen buffer-based base: no CUDA/quantized checkpoint required.'''
        def __init__(self, in_features, out_features):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.bias = None
            self.register_buffer('packed_metadata', torch.zeros(1, dtype=torch.int32))
            self.register_buffer('weight', torch.randn(out_features, in_features))

        def forward(self, x):
            return nn.functional.linear(x, self.weight)

    experts = [Expert() for _ in range(3)]
    for expert in experts:
        for name in ('up_proj', 'gate_proj', 'down_proj'):
            linear = getattr(expert, name)
            original = GPTQLoraLinear(
                BufferLinear(linear.in_features, linear.out_features),
                'default', LoraConfig(r=2), r=2,
            )
            bridge = LinearBridge(name=name)
            bridge.set_original_component(original)
            setattr(expert, name, bridge)
    model = SparseModel(experts=experts)
    builder = TorchPruningGroupBuilder(model, torch.randn(2, 4, requires_grad=True))
    for expert in model.experts:
        target = expert.up_proj._original_component.lora_B['default']
        assert target in builder.DG.module2node
        group = builder.DG.get_pruning_group(target, tp.prune_linear_out_channels, [1, 3])
        assert group[0].dep.target.module is target
        assert list(group[0].idxs) == [1, 3]
    assert [expert.calls for expert in model.experts] == [2, 1, 1]
