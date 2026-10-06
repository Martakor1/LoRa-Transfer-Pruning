from pruning_lens.core.compairing_trace import CompairingTrace, ComparisonStatus, TensorComparison


def make_trace():
    return CompairingTrace(
        activations={
            "hidden_fwd": TensorComparison(ComparisonStatus.UNRESOLVED),
            "first": TensorComparison(ComparisonStatus.OK),
            "hidden_bwd": TensorComparison(ComparisonStatus.DIFF),
            "last": TensorComparison(ComparisonStatus.DIFF),
        },
        gradients={
            "gradient_only": TensorComparison(ComparisonStatus.DIFF),
            "hidden_fwd": TensorComparison(ComparisonStatus.DIFF),
            "hidden_bwd": TensorComparison(ComparisonStatus.UNRESOLVED),
            "first": TensorComparison(ComparisonStatus.NO_GRAD),
        },
        metric=TensorComparison(ComparisonStatus.OK),
    )


def test_hidden_hooks_are_absent_from_numbering_and_summary(capsys):
    make_trace().print_summary()
    output = capsys.readouterr().out
    assert "hidden" not in output
    assert "gradient_only" not in output
    assert "[00] first" in output
    assert "[01] last" in output
    assert "[02]" not in output
    assert "FIRST FWD DIFF: last; not compared successfully: 0" in output
    assert "FIRST BWD DIFF: None; not compared successfully: 1" in output
    assert "METRIC" in output


def test_show_unresolved_keeps_activation_order(capsys):
    make_trace().print_summary(hide_unresolved_hooks=False)
    output = capsys.readouterr().out
    assert "gradient_only" not in output
    for index, name in enumerate(("hidden_fwd", "first", "hidden_bwd", "last")):
        assert f"[{index:02d}] {name}" in output
    assert "FIRST FWD DIFF: hidden_bwd; not compared successfully: 1" in output
    assert "FIRST BWD DIFF: hidden_fwd; not compared successfully: 2" in output


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
