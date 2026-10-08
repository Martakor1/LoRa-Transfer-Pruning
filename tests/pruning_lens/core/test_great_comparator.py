import math

import pytest
import torch

from pruning_lens.core.activation_corruption import ActivationCorruption, RemovedChannelsDefect, ReshapeDefect
from pruning_lens.core.compairing_trace import ComparisonStatus
from pruning_lens.core.great_comparator import GreatComparator
from pruning_lens.core.pruning_trace import PruningTrace


def make_trace(tensor, gradient=None, defects=(), name="q", unresolved=False):
    return PruningTrace(
        activations={name: tensor}, gradients={name: gradient}, metric=torch.tensor(1.),
        corruptions={name: ActivationCorruption(list(defects))},
        unresolved_pruning_hooks={name} if unresolved else set(),
    )


def test_metrics_difference_and_reference_relative_tolerance():
    ref = torch.tensor([3., 0.], requires_grad=True)
    test = torch.tensor([6., 4.], requires_grad=True)
    result = GreatComparator.compare_traces(make_trace(ref), make_trace(test))
    row = result.activations["q"]
    assert row.status is ComparisonStatus.DIFF
    torch.testing.assert_close(row.difference, torch.tensor([3., 4.]))
    assert row.max_abs == 4
    assert row.mean_abs == 3.5
    assert row.rmse == pytest.approx(math.sqrt(12.5))
    assert row.abs_l2 == 5
    assert row.rel_l2 == pytest.approx(5 / 3)
    assert not row.difference.requires_grad
    assert ref.grad is test.grad is None
    assert result.gradients["q"].status is ComparisonStatus.NO_GRAD
    assert result.metric.status is ComparisonStatus.OK
    # Tolerance is relative to reference=1, not candidate=2.
    row = GreatComparator.compare_traces(
        make_trace(torch.tensor([1.])), make_trace(torch.tensor([2.])), atol=0, rtol=.75,
    ).activations["q"]
    assert row.status == "DIFF"


def test_restore_both_layouts_and_compare_full_gradients():
    compact = torch.arange(8.).reshape(1, 2, 4)
    removed = RemovedChannelsDefect(torch.tensor([1, 4]))
    full = removed.restore_activation(compact)
    ref = make_trace(
        compact.reshape(1, 2, 2, 2), torch.ones(1, 2, 2, 2),
        [removed, ReshapeDefect((1, 2, 4))],
    )
    candidate_grad = torch.ones_like(full)
    candidate_grad[..., [1, 4]] = 0
    candidate = make_trace(
        full.reshape(1, 2, 2, 3), candidate_grad.reshape(1, 2, 2, 3),
        [ReshapeDefect((1, 2, 6))],
    )
    result = GreatComparator.compare_traces(ref, candidate)
    assert result.activations["q"].status == result.gradients["q"].status == "OK"
    gradient = result.gradients["q"]
    assert gradient.compared_elements == 12
    assert gradient.max_abs == 0
    assert gradient.difference.eq(0).all()
    assert gradient.difference.device.type == "cpu"
    candidate.gradients["q"].reshape(1, 2, 6)[..., [1, 4]] = 17
    full_result = GreatComparator.compare_traces(ref, candidate)
    assert full_result.gradients["q"].status == "DIFF"
    assert full_result.gradients["q"].max_abs == 17
    # A missing activation mask must also produce a difference.
    candidate.activations["q"].reshape(1, 2, 6)[..., 1] = 10
    assert GreatComparator.compare_traces(ref, candidate).activations["q"].status == "DIFF"


def test_gradients_include_removed_coordinates_for_two_structural_traces():
    ref = make_trace(torch.tensor([1., 2.]), torch.tensor([7., 8.]), [RemovedChannelsDefect(torch.tensor([0]))])
    test = make_trace(torch.tensor([3., 2.]), torch.tensor([99., 8.]), [RemovedChannelsDefect(torch.tensor([1]))])
    row = GreatComparator.compare_traces(ref, test).gradients["q"]
    assert row.status == "DIFF"
    assert row.compared_elements == 3
    torch.testing.assert_close(row.difference, torch.tensor([99., -7., 0.]))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_dtypes_and_optional_storage(dtype):
    ref = make_trace(torch.tensor([1., 2.], dtype=dtype))
    candidate = make_trace(torch.tensor([1., 3.], dtype=dtype))
    result = GreatComparator.compare_traces(ref, candidate, compute_gradient=False, store_difference=False)
    assert result.gradients == {}
    assert result.activations["q"].difference is None
    assert result.activations["q"].max_abs == 1


@pytest.mark.parametrize("case,expected", [
    ("missing", "MISSING"), ("shape", "SHAPE"), ("unresolved", "UNRESOLVED"),
    ("missing_defects", "UNRESOLVED"), ("bad_indices", "RESTORE_ERROR"),
    ("nan", "NONFINITE"), ("inf", "NONFINITE"), ("empty", "EMPTY"),
])
def test_non_comparable_statuses(case, expected):
    ref = make_trace(torch.ones(2))
    candidate = make_trace(torch.ones(2))
    if case == "missing":
        candidate = make_trace(torch.ones(2), name="other")
    elif case == "shape":
        candidate.activations["q"] = torch.ones(3)
    elif case == "unresolved":
        candidate.unresolved_pruning_hooks.add("q")
    elif case == "missing_defects":
        candidate.corruptions.clear()
    elif case == "bad_indices":
        candidate.corruptions["q"].defects.append(RemovedChannelsDefect(torch.tensor([99])))
    elif case in ("nan", "inf"):
        candidate.activations["q"][0] = float(case)
    elif case == "empty":
        ref.activations["q"] = candidate.activations["q"] = torch.empty(0)
    result = GreatComparator.compare_traces(ref, candidate)
    assert result.activations["q"].status is ComparisonStatus(expected)
    assert result.activations["q"].max_abs is None
    if case == "missing":
        assert list(result.activations) == ["q", "other"]
        assert result.activations["other"].status == "MISSING"


def test_status_enum_string_representation():
    assert str(ComparisonStatus.OK) == "OK"
    assert f"{ComparisonStatus.DIFF:4s}" == "DIFF"
    assert ComparisonStatus("NO_GRAD") is ComparisonStatus.NO_GRAD


@pytest.mark.parametrize("ref_grad,test_grad", [(None, None), (None, torch.ones(2)), (torch.ones(2), None)])
def test_missing_gradient_never_passes(ref_grad, test_grad):
    result = GreatComparator.compare_traces(
        make_trace(torch.ones(2), ref_grad), make_trace(torch.ones(2), test_grad),
    )
    assert result.gradients["q"].status == "NO_GRAD"


def test_print_summary_reports_diff_and_uncompared_separately(capsys):
    result = GreatComparator.compare_traces(make_trace(torch.zeros(2)), make_trace(torch.ones(2)))
    result.print_summary()
    output = capsys.readouterr().out
    assert "FIRST FWD DIFF: q" in output
    assert "FIRST BWD DIFF: None; not compared successfully: 1" in output
    for label in ("max_abs=", "mean_abs=", "rmse=", "rel_l2=", "abs_l2=", "NO_GRAD"):
        assert label in output


@pytest.mark.parametrize("kwargs", [{"atol": -1}, {"rtol": float("nan")}])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        GreatComparator.compare_traces(make_trace(torch.ones(2)), make_trace(torch.ones(2)), **kwargs)
