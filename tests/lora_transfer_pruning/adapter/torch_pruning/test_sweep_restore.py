"""The experiment must not accumulate pruning between configurations."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch_pruning as tp


def load_sweep():
    root = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location(
        'pruning_sweep', root / 'experiments/compare_tp_and_our/greate_comparator_metrics/pruning_sweep.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 4)

    def reset_hooks(self):
        pass


def test_restore_parameters_metadata_and_only_temporary_pre_hooks():
    sweep = load_sweep()
    model = TinyModel()
    original_weight = model.linear.weight
    x = torch.randn(2, 8)
    expected = model.linear(x)
    permanent = model.linear.register_forward_pre_hook(lambda module, args: None)
    restore = sweep.ModelRestorePoint(model)
    tp.prune_linear_in_channels(model.linear, [1, 5])
    model.linear.register_forward_pre_hook(lambda module, args: (args[0] * 0,))
    restore.restore()
    assert model.linear.weight is original_weight
    assert model.linear.in_features == 8
    assert list(model.linear._forward_pre_hooks) == [permanent.id]
    torch.testing.assert_close(model.linear(x), expected, rtol=0, atol=0)


def test_grid_contains_unique_ids_and_full_factorial():
    sweep = load_sweep()
    configs = [c for model in sweep.MODELS for dtype in ('bf16', 'fp16', 'fp32')
               for c in sweep.configurations(model, dtype)]
    assert len(configs) == len({c['id'] for c in configs}) == 486
    assert {c['coverage'] for c in configs} == {'single', 'distributed', 'full'}
    assert {c['physical_gpu'] for c in configs} == {2}
    assert all(not c['compute_gradient'] for c in configs)


def test_full_grid_can_be_run_without_repeating_other_coverages():
    sweep = load_sweep()
    configs = [c for model in sweep.MODELS for dtype in ('bf16', 'fp16', 'fp32')
               for c in sweep.configurations(model, dtype, ('full',))]
    assert len(configs) == 162
    assert {c['coverage'] for c in configs} == {'full'}


@pytest.mark.parametrize('model_name', ['llama', 'gemma', 'deepseek'])
def test_full_task_covers_all_attention_layers_and_supported_mlps(model_name):
    sweep = load_sweep()
    blocks = [SimpleNamespace(attn=SimpleNamespace(_original_component=SimpleNamespace(
        head_dim=8, qk_nope_head_dim=4, qk_rope_head_dim=4, kv_lora_rank=8,
        is_kv_shared_layer=(i == 3), store_full_length_kv=(i == 1),
        layer_type='full_attention',
    ))) for i in range(4)]
    bridge = SimpleNamespace(blocks=blocks, get_submodule=lambda name: SimpleNamespace(
        _original_component=torch.nn.Linear(8, 16)))
    cfg = dict(model=model_name, coverage='full', ratio=.5, seed=42, mlp=True)
    task, requested, affected, mlp_layers = sweep.make_task(bridge, cfg)
    assert requested == affected == [0, 1, 2, 3]
    assert mlp_layers == ([0] if model_name == 'deepseek' else [0, 1, 2, 3])
    attn_names = [name for name in task.data if '.attn.' in name]
    assert len(attn_names) == (3 if model_name == 'gemma' else 4)
    if model_name == 'gemma':
        assert 'blocks.1.attn.q' in attn_names
        assert 'blocks.3.attn.q' not in attn_names  # covered by the same KV group


@pytest.mark.parametrize('percent,expected', [
    (10, 12), (20, 26), (30, 38), (40, 52), (50, 64),
    (60, 76), (70, 90), (80, 102), (90, 116),
])
def test_deepseek_nope_rounding(percent, expected):
    sweep = load_sweep()
    count = sweep.deepseek_nope_pruned_count(128, percent / 100)
    assert count == expected
    assert (128 - count) % 2 == 0
    assert abs(count - 128 * percent / 100) <= 1


def test_deepseek_nope_rounding_ties_and_boundaries():
    sweep = load_sweep()
    assert sweep.deepseek_nope_pruned_count(128, 25 / 128) == 26
    assert sweep.deepseek_nope_pruned_count(128, 0) == 0
    assert sweep.deepseek_nope_pruned_count(128, .999) == 126
