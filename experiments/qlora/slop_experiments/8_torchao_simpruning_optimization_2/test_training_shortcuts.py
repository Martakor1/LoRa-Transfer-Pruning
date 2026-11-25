"""CPU regression checks for the experiment's opt-in traversal caches."""
from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import torch
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "lora_transfer_pruning").is_dir())))
from experiments.qlora.torchao_exploration.optimize_qwen_simpruning import cache_flop_count, cache_parameter_iteration, skip_redundant_train


def test_parameter_cache_preserves_aliases_order_clipping_and_zero_grad():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    model.alias = model[0]
    expected = tuple(model.parameters())
    root_only = tuple(model.parameters(recurse=False))
    for p in expected:
        p.grad = torch.ones_like(p)
    original = cache_parameter_iteration(model)
    assert tuple(map(id, model.parameters())) == tuple(map(id, expected))
    assert tuple(map(id, model.parameters(recurse=False))) == tuple(map(id, root_only))
    assert tuple(map(id, model.parameters())) == tuple(map(id, expected))  # Fresh iterator.
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    assert torch.isfinite(norm)
    model.zero_grad(set_to_none=True)
    assert all(p.grad is None for p in expected)
    object.__setattr__(model, 'parameters', original)
    assert tuple(map(id, model.parameters())) == tuple(map(id, expected))


def test_flops_cache_keeps_variable_batch_formula():
    calls = []
    def num_parameters(exclude_embeddings):
        assert exclude_embeddings
        calls.append(True)
        return 17
    model = SimpleNamespace(num_parameters=num_parameters, main_input_name='input_ids')
    def original(batch):
        return 6 * batch['input_ids'].numel() * model.num_parameters(True) if 'input_ids' in batch else 0
    trainer = SimpleNamespace(model=model, floating_point_ops=original)
    saved = cache_flop_count(trainer, {'input_ids': torch.zeros(2, 4)})
    initial_calls = len(calls)
    assert trainer.floating_point_ops({'input_ids': torch.zeros(3, 5)}) == 6 * 15 * 17
    assert trainer.floating_point_ops({}) == 0
    assert len(calls) == initial_calls
    assert saved is original


def test_mode_shortcut_keeps_real_transitions():
    model = torch.nn.Sequential(torch.nn.Dropout(), torch.nn.Linear(2, 2))
    original = skip_redundant_train(model)
    assert model.train() is model
    model.eval()
    assert all(not module.training for module in model.modules())
    model.train()
    assert all(module.training for module in model.modules())
    object.__setattr__(model, 'train', original)
