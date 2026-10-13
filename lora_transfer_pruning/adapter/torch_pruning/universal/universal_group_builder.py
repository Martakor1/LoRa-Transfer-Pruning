from typing import Callable, cast

import peft
from torch import nn
import torch_pruning as tp
import torch
from transformers import PreTrainedConfig

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge
from .universal_structural_setup import UniversalStructuralSetup
from lora_transfer_pruning.adapter.torch_pruning.index_utils import IndexUtils

class UniversalGroupBuilder:
            
    @staticmethod
    def prepare_local_attn_head_indices(
        idxs: torch.Tensor | float,
        head_dim: int,
        has_rope: bool,
        device: torch.device,
    ) -> torch.Tensor:
        '''Prepare local channels before repeating them over attention heads.

        Use the actual layer head_dim: Gemma full and sliding attention differ.
        '''
        if isinstance(idxs, (float, int)):
            idxs = IndexUtils.convert_head_idx_fraction_to_idxs(idxs, head_dim, 2 if has_rope else 1, device)

        if has_rope:
            idxs = IndexUtils.close_rope_pairs(idxs, head_dim)

        if len(idxs) >= head_dim:
            raise ValueError('Cannot remove the entire attention head')
        return idxs
    
    @staticmethod
    def get_correct_pruning_group_and_structural_setup_for_attn(model_bridge: TransformerBridge, 
                                                            bridge_name: str, 
                                                            idxs: torch.Tensor | float, 
                                                            device: torch.device
                                                            ) -> tuple[torch.Tensor, Callable[[], None]]:
        '''Creates torch pruning group and fixes indices for k and v modules in attn (for example for kv_repeat).

        For rope mirrors indices.

        Also returns setup function to apply later. It change static fields related to module for torch pruning shapes compatability (self_attn.head_dim for example).
        If nothing to change returns no-op function.'''

        attn_module = model_bridge.get_submodule(bridge_name[:-2])
        assert attn_module._original_component.config is not None #type: ignore
        attn_config = cast(PreTrainedConfig, attn_module._original_component.config) #type: ignore
        head_dim = attn_config.head_dim
        idxs = UniversalGroupBuilder.prepare_local_attn_head_indices(
            idxs, head_dim, hasattr(model_bridge, 'rotary_emb'), device)

        repeating_heads = cast(int, attn_config.num_attention_heads)
        idxs = IndexUtils.manually_indices_repeating(
            repeating_heads,
            head_dim,
            idxs
        )

        setup_function = UniversalStructuralSetup.get_setup_function_for_ordinary_attn(attn_module._original_component, attn_module, idxs)
   
        return idxs, setup_function
    
    @staticmethod
    def replace_linear_indices(
        group,
        module: LinearBridge | nn.Parameter,
        new_idxs,
        prune_out,
        DG: tp.DependencyGraph,
    ):
        '''Replaces channels to prune in group for concrete module'''
        new_idxs = sorted(set(map(int, new_idxs)))
        matched_items = set()
        
        potential_deps = set([module]) #if nn.Parameter
        if (isinstance(module, LinearBridge)):
            potential_deps = set([module._original_component]) # suppose ordinary nn.Linear
            if (isinstance(module._original_component, peft.tuners.lora.Linear)):
                active_adapters = module._original_component.active_adapters
                potential_deps = set([module._original_component.base_layer])
                if (prune_out):
                    for adapter_name in active_adapters:
                        potential_deps.add(module._original_component.lora_B[adapter_name])
                else:
                    for adapter_name in active_adapters:
                        potential_deps.add(module._original_component.lora_A[adapter_name])

        for i, (dep, _) in enumerate(group):
            if dep.target.module not in potential_deps:
                continue
            
            if prune_out:
                correct_handler = DG.is_out_channel_pruning_fn(dep.handler)
            else:
                correct_handler = DG.is_in_channel_pruning_fn(dep.handler)

            if not correct_handler:
                continue
            
            if dep.target.module in matched_items:
                raise RuntimeError(f"Module {dep.target.module} with handler {dep.handler} found twice in group.")

            group[i] = tp._helpers.GroupItem(
                dep=dep,
                idxs=new_idxs,
            )

            matched_items.add(dep.target.module)

        assert len(matched_items) == len(potential_deps), (
            f"Expected {len(potential_deps)} group items for {module}, "
            f"found {matched_items}"
        )
