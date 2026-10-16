import pytest
import torch
from transformer_lens.conversion_utils.conversion_steps.rearrange_tensor_conversion import RearrangeTensorConversion

from lora_transfer_pruning.core.prune_task_type import GroupPruneTask, ModelPruneTask
from lora_transfer_pruning.usecase.local_pruning import LocalPruning
from pruning_lens.adapter.transfer_pruning.transfer_pruning_tracer import TransferPruningTracer
from pruning_lens.usecase.compare_pruned_models_usecase import ComparePrunedModelsUsecase


@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("direction", ["rows", "cols"])
@pytest.mark.parametrize("indices", [[1, 4], .5])
def test_one_plan_sp_tp_roundtrip_and_hook_cleanup(tiny_pruning_model, monkeypatch, lora, direction, indices):
    torch.manual_seed(7)
    model = tiny_pruning_model(lora=lora)
    model.train()
    model.last.eval()  # Preserve mixed training flags, not only model.training.
    original_modes = [module.training for module in model.modules()]
    tokens = torch.randn(2, 3, 6)
    model.first.hook_out.add_hook(lambda tensor, hook: tensor * 1.25)
    task = ModelPruneTask({"first": GroupPruneTask(
        cols=indices if direction == "cols" else None,
        rows=indices if direction == "rows" else None,
    )})
    build = LocalPruning.get_torch_pruning_groups_and_structural_setups
    built_plans = []

    def record_plan(self, prune_task):
        plan = build(self, prune_task)
        built_plans.append(plan)
        return plan

    monkeypatch.setattr(LocalPruning, "get_torch_pruning_groups_and_structural_setups", record_plan)
    result = ComparePrunedModelsUsecase.compare_transfer_and_torch_prunings(
        model, tokens, task, print_summary=False,
    )
    assert len(built_plans) == 1
    assert all(row.status == "OK" for row in result.activations.values())
    assert all(row.status == "OK" for row in result.gradients.values())
    assert result.metric.status == "OK"
    assert [module.training for module in model.modules()] == original_modes
    assert not model.first.hook_out.fwd_hooks
    for hook in model.hook_dict.values():
        assert not hook.bwd_hooks
        if hook is not model.first.hook_out:
            assert not hook.fwd_hooks
    width = 4 if isinstance(indices, list) else 3
    expected = (width, 6) if direction == "rows" else (6, width)
    assert model.first.original_component.weight.shape == expected


def test_capture_with_conversion_and_explicit_upstream(tiny_pruning_model):
    model = tiny_pruning_model()
    # Use the instrumentor's Q-projection wrapper for flattened head masks.
    model.first.name = "q_proj"
    model.first.hook_out.hook_conversion = RearrangeTensorConversion("b s (h d) -> b s h d", h=2)
    result = ComparePrunedModelsUsecase.compare_transfer_and_torch_prunings(
        model, torch.randn(2, 3, 6),
        ModelPruneTask({"first": GroupPruneTask(cols=None, rows=[1, 4])}),
        name_filter=["stem.hook_out", "first.hook_out", "last.hook_in"],
        print_summary=False,
    )
    row = result.activations["first.hook_out"]
    assert row.reference_shape == (2, 3, 2, 2)
    assert row.candidate_shape == (2, 3, 2, 3)
    assert row.reference_restored_shape == row.candidate_restored_shape == (2, 3, 6)
    assert all(row.status == "OK" for row in result.activations.values())
    assert all(row.status == "OK" for row in result.gradients.values())


def test_activation_only_summary_and_no_difference_storage(tiny_pruning_model, capsys):
    result = ComparePrunedModelsUsecase.compare_transfer_and_torch_prunings(
        tiny_pruning_model(), torch.randn(2, 3, 6),
        ModelPruneTask({"first": GroupPruneTask(cols=None, rows=[1, 4])}),
        compute_gradient=False, store_difference=False,
    )
    assert result.gradients == {}
    assert all(row.difference is None for row in result.activations.values())
    output = capsys.readouterr().out
    assert "FIRST FWD DIFF: None" in output
    assert "BWD" not in output


@pytest.mark.parametrize("fail_during_prepare", [False, True])
def test_sp_failure_resets_hooks(tiny_pruning_model, monkeypatch, fail_during_prepare):
    model = tiny_pruning_model()
    model.first.hook_out.add_hook(lambda tensor, hook: None)
    if fail_during_prepare:
        original = LocalPruning.prepare_model_to_transfer_pruning_from_groups

        def fail(self, *args, **kwargs):
            original(self, *args, **kwargs)
            raise RuntimeError("intentional capture failure")

        monkeypatch.setattr(LocalPruning, "prepare_model_to_transfer_pruning_from_groups", fail)
    else:
        def fail(*args, **kwargs):
            raise RuntimeError("intentional capture failure")

        monkeypatch.setattr(TransferPruningTracer, "capture_hookpoints_activations_with_defects", fail)

    with pytest.raises(RuntimeError, match="intentional capture failure"):
        ComparePrunedModelsUsecase.compare_transfer_and_torch_prunings(
            model, torch.randn(2, 3, 6),
            ModelPruneTask({"first": GroupPruneTask(cols=None, rows=[1, 4])}),
        )
    assert model.training
    assert not model.first.hook_out.fwd_hooks
    assert model.first.original_component.weight.shape == (6, 6)
    assert all(not hook.bwd_hooks for hook in model.hook_dict.values())


def test_overlapping_groups_rejected_before_pruning(tiny_pruning_model):
    model = tiny_pruning_model()
    task = ModelPruneTask({
        "first": GroupPruneTask(cols=None, rows=[1, 4]),
        "last": GroupPruneTask(cols=[1, 4], rows=None),
    })
    with pytest.raises(ValueError, match="Repeated boundary"):
        ComparePrunedModelsUsecase.compare_transfer_and_torch_prunings(model, torch.randn(2, 3, 6), task)
    assert model.first.original_component.weight.shape == (6, 6)
    assert not model.first.hook_out.fwd_hooks


def test_layer_filter_respects_layer_boundaries(monkeypatch):
    captured = {}
    sentinel = object()

    def compare(*args, **kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(ComparePrunedModelsUsecase, "compare_transfer_and_torch_prunings", compare)
    result = ComparePrunedModelsUsecase.compare_transfer_and_torch_prunings_on_layers(
        None, torch.ones(1), ModelPruneTask({}), [1, 3], compute_gradient=False,
    )
    assert result is sentinel
    predicate = captured["name_filter"]
    assert predicate("blocks.1.attn.hook_in")
    assert predicate("blocks.3.mlp.hook_out")
    assert not predicate("blocks.10.attn.hook_in")
    assert not predicate("blocks.2.attn.hook_in")
    assert captured["compute_gradient"] is False
