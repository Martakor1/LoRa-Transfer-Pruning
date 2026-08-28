from typing import Callable, cast

import torch_pruning as tp
import torch
from transformers import PreTrainedConfig

from transformer_lens.model_bridge.bridge import TransformerBridge
from .universal_structural_setup import UniversalStructuralSetup
from lora_transfer_pruning.adapter.torch_pruning.index_utils import IndexUtils

class UniversalGroupBuilder:
            
    @staticmethod
    def _convert_head_idx_fraction_to_idxs(idxs: float,
                                      head_dim,
                                      rope_denominator: int,
                                      device: torch.device
                                      ) -> torch.Tensor:
        num_rows_to_prune = int(head_dim // rope_denominator * idxs)
        idxs_converted = torch.randperm(head_dim // rope_denominator, device=device)[:num_rows_to_prune]
        return idxs_converted
    
    @staticmethod
    def get_correct_pruning_group_and_structural_setup_for_attn(model_bridge: TransformerBridge, 
                                                            tp_pruning_function: Callable, 
                                                            bridge_name: str, 
                                                            idxs: torch.Tensor | float, 
                                                            device: torch.device
                                                            ) -> tuple[torch.Tensor, Callable[[], None]]:
        '''Creates torch pruning group and fixes indices for k and v modules in attn (for example for kv_repeat).

        For rope mirrors indices.

        Also returns setup function to apply later. It change static fields related to module for torch pruning shapes compatability (self_attn.head_dim for example).
        If nothing to change returns no-op function.'''
        proj_name = bridge_name[-2:]
        if (proj_name == '.q' and tp_pruning_function == tp.prune_linear_out_channels):

            attn_module = model_bridge.get_submodule(bridge_name[:-2])
            assert attn_module._original_component.config is not None #type: ignore
            attn_config = cast(PreTrainedConfig, attn_module._original_component.config) #type: ignore
            rope_denominator = 1
            #treat it is sign of RoPE (so we need to mirror indices). But it is not 100%?
                                                            
            if (hasattr(model_bridge, 'rotary_emb')):
                rope_denominator = 2
            
            if (isinstance(idxs, float | int)):
                idxs = UniversalGroupBuilder._convert_head_idx_fraction_to_idxs(idxs,
                                                        attn_config.head_dim,
                                                        rope_denominator,
                                                        device
                                                        )

            if (hasattr(model_bridge, 'rotary_emb')):
                idxs = IndexUtils.close_rope_pairs(idxs, attn_config.head_dim)

            repeating_heads = cast(int, attn_config.num_attention_heads)
            idxs = IndexUtils.manually_indices_repeating(
                repeating_heads,
                cast(int, attn_config.head_dim),
                idxs
            )

            setup_function = UniversalStructuralSetup.get_setup_function_for_ordinary_attn(attn_module._original_component, attn_module, idxs)

        else:
            raise NotImplementedError(f"Pruning for {bridge_name} with {tp_pruning_function} is not implemented yet. For attn module q,k,v and o linked. But torch_pruning can find group correctly only for q module.")
           
        return idxs, setup_function
    
    @staticmethod
    def replace_linear_indices(
        group,
        module,
        new_idxs,
        prune_out,
        DG: tp.DependencyGraph,
    ):
        '''Replaces channels to prune in group for concrete module'''
        new_idxs = sorted(set(map(int, new_idxs)))
        matched_items = []

        for i, (dep, _) in enumerate(group):
            if dep.target.module is not module:
                continue

            if prune_out:
                correct_handler = DG.is_out_channel_pruning_fn(
                    dep.handler
                )
            else:
                correct_handler = DG.is_in_channel_pruning_fn(
                    dep.handler
                )

            if not correct_handler:
                continue

            group[i] = tp._helpers.GroupItem(
                dep=dep,
                idxs=new_idxs,
            )

            matched_items.append(i)

        assert len(matched_items) == 1, (
            f"Expected one group item for {module}, "
            f"found {matched_items}"
        )