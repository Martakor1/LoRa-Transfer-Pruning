"""Automatic and explicit hook selections must obey the same filters."""

import pytest
import torch

from lora_transfer_pruning.core.prune_task_type import GroupPruneTask, ModelPruneTask
from lora_transfer_pruning.usecase.local_pruning import LocalPruning
from pruning_lens.adapter.hook_names_filter import HookNamesFilter


@pytest.fixture
def model_and_groups(tiny_pruning_model, monkeypatch):
    model = tiny_pruning_model()
    local = LocalPruning(model, torch.randn(1, 2, 6))
    plan = local.get_torch_pruning_groups_and_structural_setups(
        ModelPruneTask({"first": GroupPruneTask(cols=None, rows=[1])})
    )
    # The tiny HookedRootModule has no bridge configuration gates.
    monkeypatch.setattr(model, "_gated_hook_reason", lambda name: None, raising=False)
    return model, plan.groups


@pytest.mark.parametrize("automatic", [True, False])
def test_disabled_hooks_are_filtered(model_and_groups, monkeypatch, automatic):
    model, groups = model_and_groups
    names = None if automatic else ["first.hook_out", "last.hook_in"]
    monkeypatch.setattr(model, "_gated_hook_reason", lambda name: (
        "use_projection" if name == "first.hook_out" else None
    ))
    with pytest.warns(UserWarning, match="Skipping disabled capture hooks: first.hook_out"):
        selected = HookNamesFilter.filter(model, groups, names)
    assert selected == ["last.hook_in"]


@pytest.mark.parametrize("automatic", [True, False])
def test_gradient_filter_is_applied_and_rejects_empty_selection(model_and_groups, automatic):
    model, groups = model_and_groups
    names = None if automatic else ["first.hook_out", "last.hook_in"]
    model.last.requires_grad_(False)
    assert HookNamesFilter.filter(model, groups, names, only_with_grad=True) == ["first.hook_out"]
    model.first.requires_grad_(False)
    with pytest.raises(ValueError, match="No hook points selected for gradient capture"):
        HookNamesFilter.filter(model, groups, names, only_with_grad=True)
    # Activation-only capture does not apply the gradient heuristic.
    assert HookNamesFilter.filter(model, groups, names) == ["first.hook_out", "last.hook_in"]


def test_empty_automatic_selection_is_rejected(model_and_groups):
    model, _ = model_and_groups
    with pytest.raises(ValueError, match="No hook points selected"):
        HookNamesFilter.filter(model, [], None)


def test_all_automatic_hooks_disabled_is_rejected(model_and_groups, monkeypatch):
    model, groups = model_and_groups
    monkeypatch.setattr(model, "_gated_hook_reason", lambda name: "use_projection")
    with pytest.raises(ValueError, match="No enabled hook points selected"):
        HookNamesFilter.filter(model, groups, None)
