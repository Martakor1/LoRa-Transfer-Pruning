"""Tests for pruning_lens.usecase.prune_and_capture_usecase."""

from copy import deepcopy

import pytest
import torch

from lora_transfer_pruning.core.prune_task_type import GroupPruneTask, ModelPruneTask
from pruning_lens.core.pruning_trace import PruningTrace
from pruning_lens.usecase.prune_and_capture_usecase import PruneAndCaptureUsecase


@pytest.mark.parametrize("compute_gradient", [False, True])
@pytest.mark.parametrize("direction", ["rows", "cols"])
@pytest.mark.parametrize("lora", [False, True])
def test_real_local_pruning_traces_match(compute_gradient, direction, lora, tiny_pruning_model):
    torch.manual_seed(42)
    sp_model = tiny_pruning_model(lora=lora)
    tp_model = deepcopy(sp_model)
    tokens = torch.randn(2, 3, 6)
    task = ModelPruneTask({"first": GroupPruneTask(
        cols=[1, 4] if direction == "cols" else None,
        rows=[1, 4] if direction == "rows" else None,
    )})

    sp_trace = PruneAndCaptureUsecase.get_transfer_pruning_trace(
        sp_model, tokens, task, compute_gradient=compute_gradient,
    )
    tp_trace = PruneAndCaptureUsecase.get_torch_pruning_trace(
        tp_model, tokens, task, compute_gradient=compute_gradient,
    )

    expected = (
        {"first.hook_out", "last.hook_in"} if direction == "rows"
        else {"stem.hook_out", "first.hook_in"}
    )
    assert isinstance(sp_trace, PruningTrace)
    assert isinstance(tp_trace, PruningTrace)
    assert set(sp_trace.activations) == set(tp_trace.activations) == expected
    assert not tp_trace.unresolved_pruning_hooks
    torch.testing.assert_close(sp_trace.metric, tp_trace.metric)
    assert sp_model.first.original_component.weight.shape == (6, 6)
    assert tp_model.first.original_component.weight.shape == (
        (4, 6) if direction == "rows" else (6, 4)
    )
    for name in expected:
        restored = tp_trace.corruptions[name].restore_activation(tp_trace.activations[name])
        torch.testing.assert_close(restored, sp_trace.activations[name])
        if compute_gradient:
            # Removed coordinates can have nonzero SP gradients *after* masking.
            # Only gradients in retained coordinates are equivalent here.
            restored_grad = tp_trace.corruptions[name].restore_activation(tp_trace.gradients[name])
            torch.testing.assert_close(
                restored_grad[..., [0, 2, 3, 5]],
                sp_trace.gradients[name][..., [0, 2, 3, 5]],
            )
        else:
            assert sp_trace.gradients[name] is None
            assert tp_trace.gradients[name] is None


@pytest.mark.parametrize("method", [
    PruneAndCaptureUsecase.get_transfer_pruning_trace,
    PruneAndCaptureUsecase.get_torch_pruning_trace,
])
@pytest.mark.parametrize("names_filter", [
    "last.hook_out",
    ["last.hook_out"],
    lambda name: name == "last.hook_out",
])
def test_explicit_filter_is_preserved(method, names_filter, tiny_pruning_model):
    model = tiny_pruning_model()
    task = ModelPruneTask({"first": GroupPruneTask(cols=None, rows=[1, 4])})
    trace = method(model, torch.randn(2, 3, 6), task,
                   compute_gradient=False, names_filter=names_filter)
    assert set(trace.activations) == {"last.hook_out"}


@pytest.mark.parametrize("method", [
    PruneAndCaptureUsecase.get_transfer_pruning_trace,
    PruneAndCaptureUsecase.get_torch_pruning_trace,
])
def test_empty_task_does_not_use_tl_default_hooks(method, tiny_pruning_model):
    with pytest.raises(ValueError, match="No hook points selected"):
        method(tiny_pruning_model(), torch.randn(2, 3, 6), ModelPruneTask({}),
               compute_gradient=False)


@pytest.mark.parametrize("method", [
    PruneAndCaptureUsecase.get_transfer_pruning_trace,
    PruneAndCaptureUsecase.get_torch_pruning_trace,
])
def test_empty_filter_fails_before_applying_pruning(method, tiny_pruning_model):
    model = tiny_pruning_model()
    task = ModelPruneTask({"first": GroupPruneTask(cols=None, rows=[1, 4])})
    with pytest.raises(ValueError, match="No hook points selected"):
        method(model, torch.randn(2, 3, 6), task, names_filter=[])
    assert model.first.original_component.weight.shape == (6, 6)
    assert not model.first.hook_out.fwd_hooks


def test_structural_setup_runs_before_pruning_and_capture(monkeypatch, tiny_pruning_model):
    from pruning_lens.usecase import prune_and_capture_usecase
    from lora_transfer_pruning.usecase.prune_task_plan import PruneTaskPlan

    events = []
    model = tiny_pruning_model()
    tokens = torch.randn(2, 3, 6)
    task = ModelPruneTask({})

    class Group:
        def prune(self):
            events.append("prune")

    group = Group()
    plan = PruneTaskPlan(None, [group], [lambda: events.append("setup")])

    class LocalPruning:
        def __init__(self, model, tokens):
            pass

        def get_torch_pruning_groups_and_structural_setups(self, prune_task):
            assert prune_task is task
            return plan

    sentinel = object()

    def capture(actual_model, actual_tokens, **kwargs):
        events.append("capture")
        assert actual_model is model
        assert actual_tokens is tokens
        assert kwargs == dict(names_filter=["last.hook_out"], compute_gradient=False, groups=[group])
        return sentinel

    monkeypatch.setattr(prune_and_capture_usecase, "LocalPruning", LocalPruning)
    monkeypatch.setattr(prune_and_capture_usecase.TorchPruningTracer,
                        "capture_hookpoints_activations_with_defects", capture)
    assert PruneAndCaptureUsecase.get_torch_pruning_trace(
        model, tokens, task, compute_gradient=False, names_filter=["last.hook_out"],
    ) is sentinel
    assert events == ["setup", "prune", "capture"]
