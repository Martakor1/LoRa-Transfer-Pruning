import torch

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge


class PruningInstrumentor:
    """
    A class that provides methods to instrument a model for pruning.
    """

    @staticmethod
    def prepare_linear_for_pruning(module: LinearBridge, pruned_cols: torch.Tensor, pruned_rows: torch.Tensor):
        """
        Prepares the linear module for pruning by instrumenting it with the necessary hooks.

        Args:
            module (LinearBridge): The TransformerLens module to be instrumented for pruning.
            columns (torch.Tensor): The columns to be pruned.
            rows (torch.Tensor): The rows to be pruned.
        """
        def hook_for_columns(tensor: torch.Tensor, hook: HookPoint):
            return PruningInstrumentor.ablate_activation_in_linear_module(tensor, pruned_cols)
        
        module.hook_in.add_hook(hook_for_columns, dir="fwd", is_permanent=True)
        
        def hook_for_rows(tensor: torch.Tensor, hook: HookPoint):
            return PruningInstrumentor.ablate_activation_in_linear_module(tensor, pruned_rows)
        
        module.hook_out.add_hook(hook_for_rows, dir="fwd", is_permanent=True)
    
    
    @staticmethod
    def ablate_activation_in_linear_module(activation: torch.Tensor, pruned_indices: torch.Tensor):
        """
        Ablates activation in place of weight columns/rows.

        Args:
            activation (torch.Tensor): The activation tensor to be ablated. (B, N, H)
        """
        return activation.index_fill(-1, pruned_indices, 0.0)