from pruning_lens.core.compairing_trace import CompairingTrace, ComparisonStatus, TensorComparison


def make_trace():
    return CompairingTrace(
        activations={
            "hidden_with_grad": TensorComparison(ComparisonStatus.UNRESOLVED),
            "first": TensorComparison(ComparisonStatus.OK),
            "hidden_no_grad": TensorComparison(ComparisonStatus.UNRESOLVED),
            "last": TensorComparison(ComparisonStatus.DIFF),
        },
        gradients={
            "hidden_with_grad": TensorComparison(ComparisonStatus.UNRESOLVED),
            "hidden_no_grad": TensorComparison(ComparisonStatus.NO_GRAD),
            "first": TensorComparison(ComparisonStatus.NO_GRAD),
            "last": TensorComparison(ComparisonStatus.DIFF),
        },
        metric=TensorComparison(ComparisonStatus.OK),
    )


def test_hidden_hooks_are_absent_from_numbering_and_summary(capsys):
    make_trace().print_summary()
    output = capsys.readouterr().out
    assert "hidden" not in output
    assert "[00] first" in output
    assert "[01] last" in output
    assert "[02]" not in output
    assert "FIRST FWD DIFF: last; not compared successfully: 0" in output
    assert "FIRST BWD DIFF: last; not compared successfully: 1" in output
    assert "METRIC" in output


def test_show_unresolved_keeps_activation_order(capsys):
    make_trace().print_summary(hide_unresolved_hooks=False)
    output = capsys.readouterr().out
    for index, name in enumerate(("hidden_with_grad", "first", "hidden_no_grad", "last")):
        assert f"[{index:02d}] {name}" in output
    assert "FIRST FWD DIFF: last; not compared successfully: 2" in output
    assert "FIRST BWD DIFF: last; not compared successfully: 3" in output


def test_all_hooks_hidden_still_prints_metric(capsys):
    CompairingTrace(
        activations={"hidden": TensorComparison(ComparisonStatus.UNRESOLVED)},
        gradients={"hidden": TensorComparison(ComparisonStatus.NO_GRAD)},
        metric=TensorComparison(ComparisonStatus.OK),
    ).print_summary()
    output = capsys.readouterr().out
    assert "hidden" not in output
    assert "FIRST" not in output
    assert "METRIC" in output
