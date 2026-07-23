from typing import cast

import torch
from torch import nn

from transformer_lens.hook_points import HookFunction, HookPoint
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge


class PruningInstrumentor:
    """
    A class that provides methods to instrument a model for pruning.
    """
    
    @staticmethod
    def prepare_linear_for_pruning_fraction(module: LinearBridge, cols_fraction: float, rows_fraction: float):
        """
        Prepares the module for pruning by instrumenting it with the necessary hooks.  

        Args:
            module (torch.nn.Module): The module to be instrumented for pruning.
            cols_fraction (float): The fraction of columns to be pruned.
            rows_fraction (float): The fraction of rows to be pruned.
        """
        original_component = cast(nn.Linear, module.original_component)
        
        # Determine the number of columns and rows to prune based on the provided fractions
        num_cols_to_prune = int(original_component.in_features) * cols_fraction
        num_rows_to_prune = int(original_component.out_features) * rows_fraction

        # Randomly select the indices of the columns and rows to prune
        pruned_cols = torch.randperm(original_component.in_features, device=original_component.weight.device)[:num_cols_to_prune]
        pruned_rows = torch.randperm(original_component.out_features, device=original_component.weight.device)[:num_rows_to_prune]

        PruningInstrumentor.prepare_linear_for_pruning(module, pruned_cols, pruned_rows)     

    @staticmethod
    def prepare_linear_for_pruning(module: LinearBridge, pruned_cols: torch.Tensor, pruned_rows: torch.Tensor):
        """
        Prepares the linear module for pruning by instrumenting it with the necessary hooks.  

        Args:
            module (LinearBridge): The TransformerLens module to be instrumented for pruning.
            columns (torch.Tensor): The columns to be pruned.
            rows (torch.Tensor): The rows to be pruned.
        """
        #see https://transformerlensorg.github.io/TransformerLens/content/model_structure.html
        
        def hook_for_columns(tensor: torch.Tensor, hook: HookPoint):
            return PruningInstrumentor.ablate_activation_in_linear_module(tensor, pruned_cols)
        
        hook_in_fn = hook_for_columns
        if (module.name == "o_proj"):
            hook_in_fn = PruningInstrumentor._flatten_heads_wrapper(hook_for_columns)
        
        module.hook_in.add_hook(hook_in_fn, dir="fwd") #backward???? TODO
        
        original_component = cast(nn.Linear, module.original_component)
        if (original_component.bias is not None):
            #restore the bias for the pruned rows (we prune only W)
            mask = torch.zeros_like(original_component.bias)
            mask[pruned_rows] = 1

            def hook_for_rows(tensor: torch.Tensor, hook: HookPoint):
                ablated_output = PruningInstrumentor.ablate_activation_in_linear_module(tensor, pruned_rows)
                return ablated_output + original_component.bias * mask
        else:
            def hook_for_rows(tensor: torch.Tensor, hook: HookPoint):
                return PruningInstrumentor.ablate_activation_in_linear_module(tensor, pruned_rows)
        
        
        hook_out_fn = hook_for_rows
        if (module.name in ["q_proj", "k_proj", "v_proj"]):
            hook_out_fn = PruningInstrumentor._flatten_heads_wrapper(hook_for_rows)

        module.hook_out.add_hook(hook_out_fn, dir="fwd")
    
    @staticmethod
    def _flatten_heads_wrapper(
        hook_fn: HookFunction,
    ) -> HookFunction:
        '''
        TransformerLens anti-reshape 4D->3D for convinient work with activation pruning.  
        For example .index_fill(-1, pruned_indices, 0.0) works only with 3D tensors,
        but for q_proj AttentionBridge creates hook_conversion 3D->4D inside every hook_out.
        We should revoke that conversion.  
        Should work only on torch.views.
        '''
        
        def wrappedHook(tensor: torch.Tensor, hook: HookPoint) -> torch.Tensor:
            original_shape = tensor.shape
            # [B, S, H, D] → [B, S, H*D]
            tensor = tensor.flatten(-2)
            result = cast(torch.Tensor, hook_fn(tensor, hook=hook))
            # [B, S, H*D] → [B, S, H, D]
            result = result.reshape(original_shape)
            return result

        return wrappedHook

    
    @staticmethod
    def ablate_activation_in_linear_module(activation: torch.Tensor, pruned_indices: torch.Tensor) -> torch.Tensor:
        """
        Ablates activation in places linked to weight columns/rows.

        Args:
            activation (torch.Tensor): The activation tensor to be ablated. (B, N, H)
        """
        new_activation = activation.index_fill(-1, pruned_indices, 0.0)
        new_activation = PruningInstrumentor._rescale_activation_after_ablation(new_activation, activation, dim=-1)
        return new_activation
    
    @staticmethod
    def _rescale_activation_after_ablation(new_activation: torch.Tensor, old_activation: torch.Tensor, dim=-1) -> torch.Tensor:
        old_norm = old_activation.norm(dim=dim, keepdim=True)
        new_activation = new_activation * (old_norm / new_activation.norm(dim=dim, keepdim=True).clamp_min(1e-8))
        return new_activation #TODO is rescale legal in case of bias (scale*(wx+b))?