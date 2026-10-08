from typing import Callable, cast

import torch
from torch import nn
import torch_pruning as tp

from transformer_lens.model_bridge.generalized_components.linear import LinearBridge


class IndexUtils:
    @staticmethod
    def manually_indices_repeating(num_heads: int, head_dim: int, pruning_indices: torch.Tensor):
        '''Repeat indices for deleting full channels in head_dim in 4D tensors of shape [..., heads, head_dim]'''
        all_indices = []
        for head_num in range(num_heads):
            all_indices.append(
                pruning_indices + head_num * head_dim)
        return torch.cat(all_indices)

    @staticmethod
    def convert_head_idx_fraction_to_idxs(idxs: float,
                                           head_dim,
                                           rope_denominator: int,
                                           device: torch.device
                                           ) -> torch.Tensor:
        num_rows_to_prune = int(head_dim // rope_denominator * idxs)
        idxs_converted = torch.randperm(
            head_dim // rope_denominator, device=device)[:num_rows_to_prune]
        return idxs_converted
    
    @staticmethod
    def convert_linear_idx_fraction_to_idxs(idxs: float, module: LinearBridge, tp_pruning_function: Callable) -> torch.Tensor:
        original_component = cast(nn.Linear, module.original_component)
        num_of_channels = original_component.out_features
        if (tp_pruning_function == tp.prune_linear_in_channels):
            num_of_channels = original_component.in_features

        num_channels_to_prune = int(num_of_channels * idxs)
        return torch.randperm(num_of_channels, device=original_component.weight.device)[:num_channels_to_prune]
    
    @staticmethod
    def close_rope_pairs(local_idxs: torch.Tensor, head_dim: int) -> torch.Tensor:
        half = head_dim // 2
        paired = torch.where(
            local_idxs < half,
            local_idxs + half,
            local_idxs - half,
        )
        return torch.unique(torch.cat([local_idxs, paired]), sorted=True)
    
    @staticmethod
    def close_complex_rope_pairs(local_idxs: torch.Tensor) -> torch.Tensor:
        '''Close indices over complex pairs (0, 1), (2, 3), ...'''
        local_idxs = local_idxs.to(dtype=torch.long)
        paired = torch.where(
            local_idxs.remainder(2) == 0,
            local_idxs + 1,  # even -> neighbour on the right
            local_idxs - 1,  # odd -> neighbour on the left
        )
        return torch.unique(torch.cat([local_idxs, paired]), sorted=True)
    
    @staticmethod
    def map_q_indices_to_kv(
        q_idxs,
        num_q_heads,
        num_kv_heads,
        head_dim,
    ):
        '''Maps indices of Q-heads to corresponding KV-heads (size 128 -> 32 for example) using GQA mapping.'''
        assert num_q_heads % num_kv_heads == 0

        repeat = num_q_heads // num_kv_heads

        mapper = tp.dependency.index_mapping._GQAIndexMapping(
            repeat=repeat,
            head_dim=head_dim,
            reverse=True,
        )

        hybrid_q_idxs = [
            tp._helpers._HybridIndex(
                idx=int(idx),
                root_idx=int(idx),
            )
            for idx in q_idxs
        ]

        mapped_hybrid_idxs = mapper(hybrid_q_idxs)  # type: ignore

        # GQA mapping is many-to-one:
        # some Q-heads maps on one KV-head.
        kv_idxs = sorted({
            int(mapped.idx)
            for mapped in mapped_hybrid_idxs
        })

        return kv_idxs