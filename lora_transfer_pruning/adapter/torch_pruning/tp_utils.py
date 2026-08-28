from typing import Callable, cast

import peft
from torch import nn
import torch_pruning as tp

from transformer_lens.model_bridge.generalized_components.linear import LinearBridge

def is_dependency_ordinary_module(dep):
    return not type(dep.layer).__module__.startswith("torch_pruning.ops")

def get_active_module_in_dep_graph_from_linear_bridge(module: LinearBridge, tp_pruning_function: Callable) -> nn.Linear:
    '''Get original component or LoRA A/B module for LinearBridge that is in torch_pruning dependency graph.'''
    if (isinstance(module._original_component, nn.Linear)):
        active_module = module._original_component
    elif (isinstance(module._original_component, peft.tuners.lora.layer.Linear)):
        if (tp_pruning_function == tp.prune_linear_in_channels):
            # TODO support different adapters
            active_module = module._original_component.lora_A.default
        elif (tp_pruning_function == tp.prune_linear_out_channels):
            active_module = module._original_component.lora_B.default
        else:
            raise ValueError(
                f"Unsupported pruning function {tp_pruning_function} for LoRA Linear module.")
    else:
        raise ValueError(
            f"Unsupported module type {type(module._original_component)} for pruning LinearBridge.")

    return cast(nn.Linear, active_module)