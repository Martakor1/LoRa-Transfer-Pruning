import torch
from torch import nn

from lora_transfer_pruning.core.pruning_instrumentor import PruningInstrumentor
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge


def test_activation_pruning_matches_pruned_linear_weight_without_bias() -> None:
    torch.manual_seed(0)

    in_features = 7
    out_features = 5
    pruned_cols = torch.tensor([1, 4], dtype=torch.long)
    pruned_rows = torch.tensor([0, 3], dtype=torch.long)

    input_tensor = torch.randn(2, 3, in_features)
    source_linear = nn.Linear(in_features, out_features, bias=False)
    random_weight = torch.randn(out_features, in_features)
    with torch.no_grad():
        source_linear.weight.copy_(random_weight)

    # Reference path: prune a copy of the weight, leaving source_linear intact.
    pruned_weight = source_linear.weight.detach().clone()
    pruned_weight[:, pruned_cols] = 0.0
    pruned_weight[pruned_rows, :] = 0.0

    weight_pruned_linear = nn.Linear(in_features, out_features, bias=False)
    print(weight_pruned_linear.in_features)
    with torch.no_grad():
        weight_pruned_linear.weight.copy_(pruned_weight)

    # Instrumented path: keep the original weight and emulate the same pruning
    # by zeroing input/output activations around the Linear module.
    activation_pruned_linear = nn.Linear(in_features, out_features, bias=False)
    with torch.no_grad():
        activation_pruned_linear.weight.copy_(source_linear.weight)

    linear_bridge = LinearBridge(name="dummy_linear")
    linear_bridge.set_original_component(activation_pruned_linear)
    PruningInstrumentor.prepare_linear_for_pruning(
        module=linear_bridge,
        pruned_cols=pruned_cols,
        pruned_rows=pruned_rows,
    )

    pruned_input = PruningInstrumentor.ablate_activation_in_linear_module(
        input_tensor,
        pruned_cols,
    )
    source_output = source_linear(pruned_input)
    weight_pruned_output = weight_pruned_linear(pruned_input)
    expected = PruningInstrumentor._rescale_activation_after_ablation(
        weight_pruned_output,
        source_output,
        dim=-1,
    )
    actual = linear_bridge(input_tensor)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        source_linear.weight,
        activation_pruned_linear.weight,
    )


def test_activation_pruning_matches_pruned_linear_weight_with_bias() -> None:
    torch.manual_seed(0)

    in_features = 7
    out_features = 5
    pruned_cols = torch.tensor([1, 4], dtype=torch.long)
    pruned_rows = torch.tensor([0, 3], dtype=torch.long)

    input_tensor = torch.randn(2, 3, in_features)
    source_linear = nn.Linear(in_features, out_features, bias=True)
    random_weight = torch.randn(out_features, in_features)
    random_bias = torch.randn(out_features)
    with torch.no_grad():
        source_linear.weight.copy_(random_weight)
        source_linear.bias.copy_(random_bias)

    # Reference path: prune a copy of the weight while preserving the bias.
    pruned_weight = source_linear.weight.detach().clone()
    pruned_weight[:, pruned_cols] = 0.0
    pruned_weight[pruned_rows, :] = 0.0

    weight_pruned_linear = nn.Linear(in_features, out_features, bias=True)
    with torch.no_grad():
        weight_pruned_linear.weight.copy_(pruned_weight)
        weight_pruned_linear.bias.copy_(source_linear.bias)

    # Instrumented path: keep the original parameters and emulate the same
    # pruning by zeroing activations while restoring bias in pruned rows.
    activation_pruned_linear = nn.Linear(in_features, out_features, bias=True)
    with torch.no_grad():
        activation_pruned_linear.weight.copy_(source_linear.weight)
        activation_pruned_linear.bias.copy_(source_linear.bias)

    linear_bridge = LinearBridge(name="dummy_linear")
    linear_bridge.set_original_component(activation_pruned_linear)
    PruningInstrumentor.prepare_linear_for_pruning(
        module=linear_bridge,
        pruned_cols=pruned_cols,
        pruned_rows=pruned_rows,
    )

    pruned_input = PruningInstrumentor.ablate_activation_in_linear_module(
        input_tensor,
        pruned_cols,
    )
    source_output = source_linear(pruned_input)
    weight_pruned_output = weight_pruned_linear(pruned_input)

    bias_mask = torch.zeros_like(source_linear.bias)
    bias_mask[pruned_rows] = 1
    pruned_rows_bias = source_linear.bias * bias_mask
    expected = PruningInstrumentor._rescale_activation_after_ablation(
        weight_pruned_output - pruned_rows_bias,
        source_output,
        dim=-1,
    ) + pruned_rows_bias
    actual = linear_bridge(input_tensor)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        source_linear.weight,
        activation_pruned_linear.weight,
    )
    torch.testing.assert_close(
        source_linear.bias,
        activation_pruned_linear.bias,
    )
