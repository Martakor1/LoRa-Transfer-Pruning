from typing import Callable, Optional, cast

import torch
from torch import nn

from transformer_lens.hook_points import HookFunction, HookPoint
from transformer_lens.model_bridge.generalized_components.base import GeneralizedComponent
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge
from transformer_lens.model_bridge.generalized_components.mla_attention import MLAAttentionBridge


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
        num_cols_to_prune = int(original_component.in_features * cols_fraction)
        num_rows_to_prune = int(original_component.out_features * rows_fraction)

        # Randomly select the indices of the columns and rows to prune
        pruned_cols = torch.randperm(original_component.in_features, device=original_component.weight.device)[:num_cols_to_prune]
        pruned_rows = torch.randperm(original_component.out_features, device=original_component.weight.device)[:num_rows_to_prune]

        PruningInstrumentor.prepare_linear_for_pruning(module, pruned_cols, pruned_rows)     

    @staticmethod
    def _make_linear_in_hook(module: LinearBridge, pruned_cols: torch.Tensor, rescale: bool):
        def hook_for_columns(tensor: torch.Tensor, hook: HookPoint):
            return PruningInstrumentor.ablate_activation_in_linear_module(tensor, pruned_cols, rescale)
        
        hook_in_fn = hook_for_columns
        if (module.name == "o_proj"):
            hook_in_fn = PruningInstrumentor._flatten_heads_wrapper(hook_for_columns)
        return hook_in_fn
    
    @staticmethod
    def _make_linear_out_hook(module: LinearBridge, pruned_rows: torch.Tensor, rescale: bool):
        original_component = cast(nn.Linear, module.original_component)
        if (original_component.bias is not None):
            #restore the bias for the pruned rows (we prune only W, bias added on pruned rows)
            mask = torch.zeros_like(original_component.bias)
            mask[pruned_rows] = 1

            def hook_for_rows(tensor: torch.Tensor, hook: HookPoint):
                ablated_output = PruningInstrumentor.ablate_activation_in_linear_module(tensor, pruned_rows, rescale)
                return ablated_output + original_component.bias * mask
        else:
            def hook_for_rows(tensor: torch.Tensor, hook: HookPoint):
                return PruningInstrumentor.ablate_activation_in_linear_module(tensor, pruned_rows, rescale)
        
        hook_out_fn = hook_for_rows
        if (module.name in ["q_proj", "k_proj", "v_proj"]):
            hook_out_fn = PruningInstrumentor._flatten_heads_wrapper(hook_for_rows)

        return hook_out_fn
    
    @staticmethod
    def _prepare_linear_for_pruning_template(module: LinearBridge, 
                                             pruned_cols: Optional[torch.Tensor], 
                                             pruned_rows: Optional[torch.Tensor], 
                                             hook_in_fn_factory: Callable[[LinearBridge, torch.Tensor, bool], HookFunction],
                                             hook_out_fn_factory: Callable[[LinearBridge, torch.Tensor, bool], HookFunction],
                                             rescale: bool = True):
        if (pruned_cols is not None and (len(pruned_cols) != 0)):
            hook_in_fn = hook_in_fn_factory(module, pruned_cols, rescale)
            module.hook_in.add_hook(hook_in_fn, dir="fwd")
        
        if (pruned_rows is not None and (len(pruned_rows) != 0)):
            hook_out_fn = hook_out_fn_factory(module, pruned_rows, rescale)
            module.hook_out.add_hook(hook_out_fn, dir="fwd")
    
    @staticmethod
    def prepare_linear_for_pruning(module: LinearBridge,
                                   pruned_cols: Optional[torch.Tensor],
                                   pruned_rows: Optional[torch.Tensor],
                                   rescale: bool = True,
                                   active_adapters_source: Optional[Callable[[], list[str]]] = None,
                                   adapter_name: Optional[str] = None):
        """
        Prepares the linear module for pruning by instrumenting it with the necessary hooks.  

        Args:
            module (LinearBridge): The TransformerLens module to be instrumented for pruning.
            columns (torch.Tensor): The columns to be pruned.
            rows (torch.Tensor): The rows to be pruned.
            active_adapters_source (Callable[[], list[str]]): A callable that returns the list of active adapters. If None, the hooks will be applied unconditionally.
            adapter_name (str): The name of the adapter (aka prune_task_name) to be used for conditional hook application. If None, the hooks will be applied unconditionally.
        """
        #see https://transformerlensorg.github.io/TransformerLens/content/model_structure.html
        if ((active_adapters_source is not None) and (adapter_name is not None)):
            PruningInstrumentor._prepare_linear_for_pruning_template(
                module,
                pruned_cols,
                pruned_rows,
                lambda m, cols, r: PruningInstrumentor._lora_hook_wrapper(PruningInstrumentor._make_linear_in_hook(m, cols, r), active_adapters_source, adapter_name),
                lambda m, rows, r: PruningInstrumentor._lora_hook_wrapper(PruningInstrumentor._make_linear_out_hook(m, rows, r), active_adapters_source, adapter_name),
                rescale
            )
        else:
            PruningInstrumentor._prepare_linear_for_pruning_template(
                module,
                pruned_cols,
                pruned_rows,
                PruningInstrumentor._make_linear_in_hook,
                PruningInstrumentor._make_linear_out_hook,
                rescale
            )
    
    @staticmethod
    def _lora_hook_wrapper(hook_fn: HookFunction, active_adapters_source: Callable[[], list[str]], lora_adapter_name: str) -> HookFunction:
        """
        Wraps a hook function to only apply it when the specified LoRA adapter is active.
        """
        def wrapped_hook(tensor: torch.Tensor, hook: HookPoint) -> torch.Tensor:
            if lora_adapter_name in active_adapters_source():
                return cast(torch.Tensor, hook_fn(tensor, hook=hook))
            else:
                return tensor

        return wrapped_hook
    
    @staticmethod
    def _flatten_heads_wrapper(
        hook_fn: HookFunction,
    ) -> HookFunction:
        '''
        TransformerLens anti-reshape 4D->3D for convenient work with activation pruning.  
        For example .index_fill(-1, pruned_indices, 0.0) works only with 3D tensors,
        but for q_proj AttentionBridge creates hook_conversion 3D->4D inside every hook_out.
        We should revoke that conversion.  
        Should work only on torch.views and don't consume additional memory.
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
    def ablate_activation_in_linear_module(activation: torch.Tensor, pruned_indices: torch.Tensor, rescale: bool) -> torch.Tensor:
        """
        Ablates activation in places linked to weight columns/rows.

        Args:
            activation (torch.Tensor): The activation tensor to be ablated. (B, N, H)
        """
        new_activation = activation.index_fill(-1, pruned_indices, 0.0)
        #is rescaling valid for many heads in one dim in qkvo TODO
        if (rescale): #TODO make two different methods for faster execution without rescale param???
            new_activation = PruningInstrumentor._rescale_activation_after_ablation(new_activation, activation, dim=-1)
        return new_activation
    
    @staticmethod
    def _rescale_activation_after_ablation(new_activation: torch.Tensor, old_activation: torch.Tensor, dim=-1) -> torch.Tensor:
        old_norm = old_activation.norm(dim=dim, keepdim=True)
        new_activation = new_activation * (old_norm / new_activation.norm(dim=dim, keepdim=True).clamp_min(1e-8))
        return new_activation #TODO is rescale legal in case of bias (scale*(wx+b))?
        
    @staticmethod
    def prepare_rms_norm_for_pruning_dynamically(rms_module: GeneralizedComponent):
        """
        Add hook for RMSNorm module with <b>dynamic</b> zero channels recognition. Out hook contains coefficient.
        The coefficient fix norm part to be calculated with pruned num of channels, not original num.
        
        NOTE: for activations near zero can be wrong because of +eps approximation.
        NOTE: give inaccurate results, if output of original RMSNorm in bfloat16 
        NOTE: works only in single-threaded model execution 

        Args:
            module (nn.Module): The RMSNorm module to be instrumented for pruning.
            pruned_indices (torch.Tensor): The indices of the weights to be pruned.
        """
        PRUNED_LEN = -1
        def get_pruned_len(tensor: torch.Tensor, hook: HookPoint):
            nonlocal PRUNED_LEN
            PRUNED_LEN = (tensor == 0.0).sum(dim=-1).view(-1)[0].item()
            return tensor
        
        rms_module.hook_in.add_hook(get_pruned_len, dir="fwd")
        
        def hook_for_rmsnorm(tensor: torch.Tensor, hook: HookPoint):
            real_shapes = tensor.shape[-1]
            pruned_shapes = real_shapes - PRUNED_LEN
            return tensor * ((pruned_shapes / real_shapes) ** 0.5) #doesnt consider +eps~1e-6 in real RMS implementations
        
        rms_module.hook_out.add_hook(hook_for_rmsnorm, dir="fwd")
        
    @staticmethod
    def prepare_mla_attention_bridge_for_pruning(attn_bridge: MLAAttentionBridge, q_indices: torch.Tensor):
        ''' Fix scaling coefficient that use scaling = self._qk_head_dim ** (-0.5) because real _qk_head_dim
        changed. Gets real dim by substracting q_indices (with nope and pe parts) from attn.qk_head_dim.
        '''
        old_qk_head_dim = attn_bridge.qk_head_dim
        new_qk_head_dim = old_qk_head_dim - (q_indices < old_qk_head_dim).sum().item()
        scale_correction = (
            old_qk_head_dim / new_qk_head_dim
        ) ** 0.5

        def hook(tensor: torch.Tensor, hook: HookPoint):
            return tensor * scale_correction
    
        attn_bridge.hook_attn_scores.add_hook(hook, dir="fwd")