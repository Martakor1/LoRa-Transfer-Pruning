from typing import Callable, cast, Iterator

import peft
from torch import nn
import torch_pruning as tp

from transformer_lens.model_bridge.generalized_components.linear import LinearBridge
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.transformer_bridge import TransformerBridge

def is_dependency_ordinary_module(dep):
    return not type(dep.layer).__module__.startswith("torch_pruning.ops")

def get_active_module_in_dep_graph_from_linear_bridge(module: LinearBridge, tp_pruning_function: Callable) -> nn.Linear:
    '''Get original component or LoRA A/B module for LinearBridge that is in torch_pruning dependency graph.'''
    if (isinstance(module._original_component, nn.Linear)):
        active_module = module._original_component
    elif (isinstance(module._original_component, peft.tuners.lora.layer.Linear)):
        adapter_name = module._original_component.active_adapters[0] #we enough with any of active adapters, all of them part of one root LinearBridge
        if (tp_pruning_function == tp.prune_linear_in_channels):
            active_module = module._original_component.lora_A[adapter_name]
        elif (tp_pruning_function == tp.prune_linear_out_channels):
            active_module = module._original_component.lora_B[adapter_name]
        else:
            raise ValueError(
                f"Unsupported pruning function {tp_pruning_function} for LoRA Linear module.")
    else:
        raise TypeError(
            f"Unsupported module type {type(module._original_component)} for pruning LinearBridge.")

    return cast(nn.Linear, active_module)

def iter_pruned_linear_hooks(
    model: TransformerBridge,
    groups: list[tp.Group],
) -> Iterator[tuple[HookPoint, list[int]]]:
    '''Resolve group operations to canonical outer linear hooks and their indices.

    Shared by automatic capture selection and TP defect collection.
    LoRA uses the active A input / B output, not internal rank boundaries.
    '''
    visited_hooks = set()
    for group in groups:
        DG = cast(tp.DependencyGraph, group._DG)
        for dep, idxs in group:
            module = dep.target.module
            if not isinstance(module, nn.Linear) or len(idxs) == 0:
                continue
            if DG.is_out_channel_pruning_fn(dep.handler):
                pruning_fn = tp.prune_linear_out_channels
            elif DG.is_in_channel_pruning_fn(dep.handler):
                pruning_fn = tp.prune_linear_in_channels
            else:
                continue

            # DG stores the module path separately from its human-readable repr.
            module_name = DG._module2name.get(module, "")
            bridge_name, separator, _ = module_name.rpartition("._original_component")
            if not separator:
                continue
            component = model.get_submodule(bridge_name)
            if not isinstance(component, LinearBridge):
                continue
            try:
                selected_module = get_active_module_in_dep_graph_from_linear_bridge(
                    component, pruning_fn
                )
            except TypeError:
                continue
            if module is not selected_module:
                continue  # Skip base/other adapters and internal LoRA-rank operations.

            hook = component.hook_in if pruning_fn == tp.prune_linear_in_channels else component.hook_out
            if hook.name in visited_hooks:
                raise ValueError(
                    "Repeated channel pruning for the same linear boundary; "
                    "sequential coordinate mappings are not supported"
                )
            visited_hooks.add(hook.name)
            
            yield hook, idxs