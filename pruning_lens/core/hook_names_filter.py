from typing import Callable, Sequence, cast
import warnings

from pruning_lens.adapter.base_tracer import BaseTracer
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge
from transformer_lens.model_bridge.transformer_bridge import TransformerBridge
import torch_pruning as tp
from lora_transfer_pruning.adapter.torch_pruning.tp_utils import get_active_module_in_dep_graph_from_linear_bridge


class HookNamesFilter:
    
    @staticmethod
    def filter_only_enabled_grad_hooks(model: TransformerBridge, names_filter: Sequence[str]) -> list[str]:
        """Filter names_filter to only linear hooks that are enabled for gradient capture.
        | Condition | What we know |
        | --- | --- |
        | The weight or bias has `requires_grad=True` | The Linear output will also have `requires_grad=True`, provided autograd is enabled. |
        | All Linear parameters are frozen | The output may still require gradients through the input. |
        | The Linear parameters are trainable | This tells us nothing about whether its **input** has `requires_grad=True`. |
        """
        requires_grad_hooks = []
        for hook_name in names_filter:
            hook = model.hook_dict[hook_name]
            bridge_name, _, hook_relative_name = cast(str, hook.name).rpartition(".")
            # if (hook_relative_name == "hook_out"): #dont know nothing for hook_in activation's grad
            
            
            #TODO hook_in needs as upstream_hook!!!!
            
            
            bridge_module = model.get_submodule(bridge_name) if bridge_name else model
            if isinstance(bridge_module, LinearBridge):
                original_module = get_active_module_in_dep_graph_from_linear_bridge(
                    bridge_module, tp.prune_linear_out_channels
                )
                if (original_module.weight.requires_grad):
                    requires_grad_hooks.append(hook_name)
        
        return requires_grad_hooks

    @staticmethod
    def filter(model: TransformerBridge, 
               groups: list[tp.Group], 
               names_filter: str | Sequence[str] | Callable[[str], bool] | None,
               only_with_grad: bool = False) -> Sequence[str]:
        """Filter names_filter to only include hooks that are present in the model's hook_dict and that are ebabled.
        If names_filter is None, create a names_filter from pruned linear modules from `groups`"""
        if names_filter is None:
            return HookNamesFilter.create_names_filter_from_pruned_linear_modules(model, groups)
        
        names_filter = HookNamesFilter.convert_to_only_models_names(model, names_filter)
        names_filter = HookNamesFilter.filter_only_enabled_hooks(model, names_filter)
        if (only_with_grad):
            names_filter = HookNamesFilter.filter_only_enabled_grad_hooks(model, names_filter)
        return names_filter
    
    @staticmethod
    def filter_only_enabled_hooks(
        model: TransformerBridge,
        names_filter: Sequence[str],
    ) -> Sequence[str]:
        """Filter enabled hooks (that will be computed in model).
        Use Bridge gating rules, including canonical names for aliased points."""

        enabled, skipped = [], []
        for name in names_filter:
            hook = model.hook_dict.get(name)
            canonical_name = (hook.name or name) if hook is not None else name
            reason = model._gated_hook_reason(canonical_name)
            if reason is None:
                enabled.append(name)
            else:
                skipped.append(f"{name} ({reason}=False)")
        if not enabled:
            raise ValueError("No enabled hook points selected for capture")
        if skipped:
            warnings.warn(
                "Skipping disabled capture hooks: " + ", ".join(skipped),
                UserWarning,
                stacklevel=3,
            )
        return enabled
    
    @staticmethod
    def convert_to_only_models_names(model: TransformerBridge, 
                                 names_filter: str | Sequence[str] | Callable[[str], bool],
                                 ) -> Sequence[str]:
        '''Filter names_filter to only include names in model.hook_dict and return only Sequence or None.
        '''
        has_hooks = False
        if isinstance(names_filter, str):
            has_hooks = names_filter in model.hook_dict
            names_filter = [names_filter]
        elif callable(names_filter):
            has_hooks = any(names_filter(name) for name in model.hook_dict)
            names_filter = [name for name in model.hook_dict if names_filter(name)]
        else:
            new_names_filter = []
            for name in names_filter:
                if name in model.hook_dict:
                    has_hooks = True
                    new_names_filter.append(name)
                
            names_filter = new_names_filter
        if not has_hooks:
            raise ValueError(
                "No hook points selected for capture. For an empty prune_task or "
                "groups without supported linear hooks, provide explicit names_filter."
            )
        return names_filter
        
    
    @staticmethod
    def create_names_filter_from_pruned_linear_modules(
        model: TransformerBridge,
        groups: list[tp.Group],
    ) -> Sequence[str]:
        '''Auto-select affected linear hooks; reject empty selection before mutation.
        '''
        return list(dict.fromkeys(
            hook.name
            for hook, _ in BaseTracer.iter_pruned_linear_hooks(model, groups)
            if hook.name is not None
        ))
