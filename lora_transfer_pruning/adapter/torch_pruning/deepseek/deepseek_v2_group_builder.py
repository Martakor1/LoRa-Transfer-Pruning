from typing import Callable, cast

import torch
from torch import nn
import torch_pruning as tp

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge
from transformer_lens.model_bridge.generalized_components.mla_attention import MLAAttentionBridge
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2Attention, DeepseekV2RMSNorm
from lora_transfer_pruning.adapter.torch_pruning.torch_pruning_group_task import TorchPruningGroupTask
from .deepseek_v2_structural_setup import DeepseekV2StructuralSetup
from lora_transfer_pruning.adapter.torch_pruning.index_utils import IndexUtils
from lora_transfer_pruning.adapter.torch_pruning.tp_utils import get_active_module_in_dep_graph_from_linear_bridge
from lora_transfer_pruning.adapter.torch_pruning.universal.universal_group_builder import UniversalGroupBuilder


class DeepseekV2GroupBuilder:

    
    @staticmethod
    def _prepare_deepseek_q_proj_indices(local_idxs: torch.Tensor, hf_attn: DeepseekV2Attention):
        '''Map local Q indices to Q, kv_b K-nope, and kv_a shared K-RoPE.'''
        num_heads = hf_attn.num_heads
        nope_dim = hf_attn.qk_nope_head_dim
        rope_dim = hf_attn.qk_rope_head_dim
        value_dim = hf_attn.v_head_dim
        kv_lora_rank = hf_attn.kv_lora_rank
        qk_head_dim = nope_dim + rope_dim

        local_idxs = torch.unique(local_idxs.to(dtype=torch.long), sorted=True)
        if torch.any(local_idxs < 0) or torch.any(local_idxs >= qk_head_dim):
            raise IndexError(
                f"DeepSeek Q indices must be local head indices in [0, {qk_head_dim}), "
                f"got {local_idxs.tolist()}."
            )

        nope_idxs = local_idxs[local_idxs < nope_dim]
        rope_idxs = local_idxs[local_idxs >= nope_dim] - nope_dim
        rope_idxs = IndexUtils.close_complex_rope_pairs(
            rope_idxs)

        q_local_idxs = torch.cat([nope_idxs, rope_idxs + nope_dim])
        q_idxs = IndexUtils.manually_indices_repeating(
            num_heads, qk_head_dim, q_local_idxs
        )
        # kv_b output per head: [K-nope | V]; V remains untouched.
        if (nope_dim != value_dim):
            raise NotImplementedError(
                f"Can't automatically map indices from k head to v, their shapes not equal ({nope_dim} != {value_dim}). Please provide kv_lora_idxs manually.")
        kv_b_idxs = IndexUtils.manually_indices_repeating(
            # nope_idxs + nope_dim]) #TODO does we want to prune v out the same as k?
            num_heads, nope_dim + value_dim, torch.cat([nope_idxs])
        )
        # kv_a output: [KV latent | shared K-RoPE]; latent remains untouched.
        kv_a_idxs = rope_idxs + kv_lora_rank
        return q_idxs, kv_b_idxs, kv_a_idxs




    @staticmethod
    def _get_correct_deepseek_attn_idxs(idxs: torch.Tensor | float, hf_attn: DeepseekV2Attention, tp_group_task: TorchPruningGroupTask, device: torch.device):
        if isinstance(idxs, float | int):
            idxs = IndexUtils.convert_head_idx_fraction_to_idxs(
                idxs,
                cast(int, hf_attn.qk_head_dim),
                1,
                device,  # will work for LoRA too without .base_layer.weight
            )

        q_idxs, kv_b_idxs, kv_a_rope_idxs = DeepseekV2GroupBuilder._prepare_deepseek_q_proj_indices(
            idxs, hf_attn
        )

        kv_lora_idxs = tp_group_task.kv_lora_idxs_deepseek
        if (isinstance(kv_lora_idxs, float | int)):
            kv_lora_idxs = IndexUtils.convert_head_idx_fraction_to_idxs(
                kv_lora_idxs,
                hf_attn.kv_lora_rank,
                1,
                device,
            )
        elif (kv_lora_idxs is None):
            kv_lora_idxs = torch.tensor([], device=device)

        if (torch.any(kv_lora_idxs < 0).item() or torch.any(kv_lora_idxs >= hf_attn.kv_lora_rank).item()):
            raise IndexError(
                f"kv_lora_idxs must be in [0, {hf_attn.kv_lora_rank}), "
                f"got {kv_lora_idxs.tolist()}."
            )
        kv_lora_idxs = torch.unique(kv_lora_idxs, sorted=True)
        kv_a_idxs = torch.unique(
            torch.cat([kv_lora_idxs, kv_a_rope_idxs]), sorted=True
        )
        return q_idxs, kv_b_idxs, kv_lora_idxs, kv_a_idxs
    
    @staticmethod
    def _replace_deepseek_mla_linear_indices(DG: tp.DependencyGraph, hf_attn: DeepseekV2Attention, group: tp.Group, kv_b_idxs, kv_lora_idxs, kv_a_idxs, o_in_idxs):        
        UniversalGroupBuilder.replace_linear_indices(
            group, cast(LinearBridge, hf_attn.kv_b_proj), kv_b_idxs, True, DG
        )
        
        UniversalGroupBuilder.replace_linear_indices(
            group, cast(LinearBridge, hf_attn.kv_b_proj), kv_lora_idxs, False, DG
        )
        UniversalGroupBuilder.replace_linear_indices(
            group, cast(LinearBridge, hf_attn.kv_a_proj_with_mqa), kv_a_idxs, True, DG
        )
        # Q/K feature coordinates disappear in attention scores. They do
        # not remove V coordinates consumed by o_proj; TP cannot infer
        # that distinction through the attention kernel. #TODO it false, if we will want to prune v_out (we do it automatically now)
        UniversalGroupBuilder.replace_linear_indices(
            group, cast(LinearBridge, hf_attn.o_proj), o_in_idxs,
            False, DG
        )

        UniversalGroupBuilder.replace_linear_indices(  # suppose dont use lora for RMSNorm
            group, cast(DeepseekV2RMSNorm, hf_attn.kv_a_layernorm._original_component).weight, kv_lora_idxs, True, DG
        )
        # TODO for q_a_layernorm also via q_lora_rank... another indices

    @staticmethod
    def get_correct_pruning_group_and_structural_setup_for_attn(
        DG: tp.DependencyGraph,
        model_bridge: TransformerBridge,
        module: LinearBridge,
        tp_group_task: TorchPruningGroupTask,
        tp_pruning_function: Callable,
        bridge_name: str,
        idxs: torch.Tensor | float,
        device: torch.device
    ) -> tuple[tp.Group, Callable[[], None]]:
        attn_module = cast(MLAAttentionBridge, model_bridge.get_submodule(
            bridge_name[:-len(".q_proj")]
        ))
        hf_attn = cast(DeepseekV2Attention,
                       attn_module._original_component)

        q_idxs, kv_b_idxs, kv_lora_idxs, kv_a_idxs = DeepseekV2GroupBuilder._get_correct_deepseek_attn_idxs(
            idxs, hf_attn, tp_group_task, device)
        o_in_idxs = torch.tensor([], device=device)

        group = DG.get_pruning_group(
            get_active_module_in_dep_graph_from_linear_bridge(
                module, tp_pruning_function),
            tp_pruning_function,
            idxs=q_idxs.tolist(),
        )

        DeepseekV2GroupBuilder._replace_deepseek_mla_linear_indices(
            DG, hf_attn, group, kv_b_idxs, kv_lora_idxs, kv_a_idxs, o_in_idxs
        )

        setup_function = DeepseekV2StructuralSetup.get_setup_function_for_deepseek_attn(
            hf_attn, attn_module, bridge_name, q_idxs, o_in_idxs, kv_a_idxs, kv_b_idxs, kv_lora_idxs
        )

        return group, setup_function
