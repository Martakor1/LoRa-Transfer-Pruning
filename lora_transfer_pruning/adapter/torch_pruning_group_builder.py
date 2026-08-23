from typing import Callable, cast

import torch
from torch import nn
from transformers import DeepseekV2Model

from transformer_lens.model_bridge.bridge import TransformerBridge
import torch_pruning as tp

from transformer_lens.model_bridge.generalized_components.linear import LinearBridge
from lora_transfer_pruning.core.constants import TRANSFORMER_LENS_ORIGINAL_COMPONENT_SUFFIX_LEN, HOOK_IN_NAME_LEN
from lora_transfer_pruning.core.prune_task_type import GroupPruneTask

class TorchPruningGroupBuilder:

    def __init__(self, 
                 model_bridge: TransformerBridge, 
                 example_input_ids: torch.Tensor, 
                 unwrapped_params: list[torch.nn.Parameter] = []):
        '''Builds a dependency graph for pruning using torch_pruning.
        
            Params:
                model_bridge: An instance of TransformerBridge that wraps the model to be pruned.
                example_input_ids: A tensor representing batches of input IDs for the model (need for torch_pruning to build dep graph by tracing the forward pass).
                unwrapped_params: params in modules, that torch_pruning can't handle or trailing params that not in modules. For example .weight in `LlamaRMSNorm`.
                    Pass it to ignore them during pruning.
        '''
        self.DG = tp.DependencyGraph().build_dependency(
                model_bridge,
                example_inputs=example_input_ids,
                forward_fn=TorchPruningGroupBuilder._trace_forward,
                ignored_params=unwrapped_params,
                unwrapped_parameters=list(zip(unwrapped_params, [0] * len(unwrapped_params))) # type: ignore
            )
        self.model_bridge = model_bridge
    
    @staticmethod
    def _trace_forward(model_bridge, input_ids):
        return model_bridge(
            input=input_ids,
            use_cache=False,
            return_type="logits",
        )
    
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
    def manually_indices_repeating(num_heads: int, head_dim: int, pruning_indices: torch.Tensor):
        '''Repeat indices for deleting full channels in head_dim in 4D tensors of shape [..., heads, head_dim]'''
        all_indices = []
        for head_num in range(num_heads):
            all_indices.append(
                pruning_indices + head_num * head_dim)
        return torch.cat(all_indices)

    # ------------------------------------------------------------------------- #
    # DeepSeek V2 MLA helpers.
    # ------------------------------------------------------------------------- #
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
    def _prepare_deepseek_q_proj_indices(local_idxs: torch.Tensor, attn_module):
        '''Map local Q indices to Q, kv_b K-nope, and kv_a shared K-RoPE.'''
        hf_attn = attn_module._original_component
        num_heads = cast(int, hf_attn.num_heads)
        nope_dim = cast(int, hf_attn.qk_nope_head_dim)
        rope_dim = cast(int, hf_attn.qk_rope_head_dim)
        value_dim = cast(int, hf_attn.v_head_dim)
        kv_lora_rank = cast(int, hf_attn.kv_lora_rank)
        qk_head_dim = nope_dim + rope_dim

        local_idxs = torch.unique(local_idxs.to(dtype=torch.long), sorted=True)
        if torch.any(local_idxs < 0) or torch.any(local_idxs >= qk_head_dim):
            raise IndexError(
                f"DeepSeek Q indices must be local head indices in [0, {qk_head_dim}), "
                f"got {local_idxs.tolist()}."
            )

        nope_idxs = local_idxs[local_idxs < nope_dim]
        rope_idxs = local_idxs[local_idxs >= nope_dim] - nope_dim
        rope_idxs = TorchPruningGroupBuilder.close_complex_rope_pairs(rope_idxs)

        q_local_idxs = torch.cat([nope_idxs, rope_idxs + nope_dim])
        q_idxs = TorchPruningGroupBuilder.manually_indices_repeating(
            num_heads, qk_head_dim, q_local_idxs
        )
        # kv_b output per head: [K-nope | V]; V remains untouched.
        if (nope_dim != value_dim):
            raise NotImplementedError(f"Can't automatically map indices from k head to v, their shapes not equal ({nope_dim} != {value_dim}). Please provide kv_lora_idxs manually.")
        kv_b_idxs = TorchPruningGroupBuilder.manually_indices_repeating(
            num_heads, nope_dim + value_dim, torch.cat([nope_idxs]) #nope_idxs + nope_dim]) #TODO does we want to prune v out the same as k?
        )
        # kv_a output: [KV latent | shared K-RoPE]; latent remains untouched.
        kv_a_idxs = rope_idxs + kv_lora_rank
        return q_idxs, kv_b_idxs, kv_a_idxs

    @staticmethod
    def _replace_deepseek_mla_linear_indices(
        group, module, new_idxs, prune_out: bool, DG
    ):
        replacement = sorted(set(map(int, new_idxs.tolist())))
        for i, (dep, _) in enumerate(group): # type: ignore
            if dep.target.module is not module:
                continue
            correct_handler = (
                DG.is_out_channel_pruning_fn(dep.handler)
                if prune_out
                else DG.is_in_channel_pruning_fn(dep.handler)
            )
            if correct_handler:
                group[i] = tp._helpers.GroupItem(dep=dep, idxs=replacement)
    # ------------------------------------------------------------------------- #


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
    def _convert_linear_idx_fraction_to_idxs(idxs: float, module: LinearBridge, tp_pruning_function: Callable) -> torch.Tensor:
        original_component = cast(nn.Linear, module.original_component)
        num_of_channels = original_component.out_features
        if (tp_pruning_function == tp.prune_linear_in_channels):
            num_of_channels = original_component.in_features
        
        num_channels_to_prune = int(num_of_channels * idxs)
        return torch.randperm(num_of_channels, device=original_component.weight.device)[:num_channels_to_prune]
                
        
    def get_correct_pruning_group(self, module: LinearBridge, tp_pruning_function: Callable, groupPruneTask: GroupPruneTask) -> tp.Group:
        '''Creates torch pruning group and fixes indices for k and v modules in attn (for example for kv_repeat).
        
        For rope mirrors indices.'''
        assert module.hook_in.name is not None
        bridge_name = module.hook_in.name[:-HOOK_IN_NAME_LEN]
        
        if (isinstance(groupPruneTask.rows, list)):
            idxs = torch.tensor(groupPruneTask.rows)
        else:
            idxs = groupPruneTask.rows
        if (tp_pruning_function == tp.prune_linear_in_channels):
            if (isinstance(groupPruneTask.cols, list)):
                idxs = torch.tensor(groupPruneTask.cols)
            else:
                idxs = groupPruneTask.cols

        # --------------------------------------------------------------------- #
        # DeepSeek V2 MLA direct-Q path.
        # --------------------------------------------------------------------- #
        if (bridge_name.endswith(".attn.q_proj")
            and tp_pruning_function == tp.prune_linear_out_channels
            and isinstance(self.model_bridge.model, DeepseekV2Model)):
            
            attn_module = self.model_bridge.get_submodule(
                bridge_name[:-len(".q_proj")]
            )
            hf_attn = attn_module._original_component

            if isinstance(idxs, float):
                idxs = self._convert_head_idx_fraction_to_idxs(
                    idxs,
                    cast(int, hf_attn.qk_head_dim),
                    1,
                    module._original_component.weight.device,
                )

            q_idxs, kv_b_idxs, kv_a_rope_idxs = self._prepare_deepseek_q_proj_indices(
                idxs, attn_module
            )
            
            kv_lora_idxs = torch.tensor([])
            if (isinstance(groupPruneTask.kv_lora_idxs_deepseek, list)):
                kv_lora_idxs = torch.as_tensor(
                    groupPruneTask.kv_lora_idxs_deepseek,
                    dtype=torch.long,
                    device=q_idxs.device,
                )
            elif(isinstance(groupPruneTask.kv_lora_idxs_deepseek, float)):
                kv_lora_idxs = self._convert_head_idx_fraction_to_idxs(
                    groupPruneTask.kv_lora_idxs_deepseek,
                    cast(int, hf_attn.kv_lora_rank),
                    1,
                    module._original_component.weight.device,
                )

            if torch.any(kv_lora_idxs < 0) or torch.any(
                kv_lora_idxs >= cast(int, hf_attn.kv_lora_rank)
            ):
                raise IndexError(
                    f"kv_lora_idxs must be in [0, {hf_attn.kv_lora_rank}), "
                    f"got {kv_lora_idxs.tolist()}."
                )
            kv_lora_idxs = torch.unique(kv_lora_idxs, sorted=True)
            kv_a_idxs = torch.unique(
                torch.cat([kv_lora_idxs, kv_a_rope_idxs]), sorted=True
            )

            group = self.DG.get_pruning_group(
                module._original_component,
                tp_pruning_function,
                idxs=q_idxs.tolist(),
            )
            self._replace_deepseek_mla_linear_indices(
                group, hf_attn.q_proj._original_component, q_idxs, True, self.DG
            )
            self._replace_deepseek_mla_linear_indices(
                group, hf_attn.kv_b_proj._original_component, kv_b_idxs, True, self.DG
            )
            self._replace_deepseek_mla_linear_indices(
                group, hf_attn.kv_b_proj._original_component, kv_lora_idxs, False, self.DG
            )
            self._replace_deepseek_mla_linear_indices(
                group, hf_attn.kv_a_proj_with_mqa._original_component, kv_a_idxs, True, self.DG
            )
            # Q/K feature coordinates disappear in attention scores. They do
            # not remove V coordinates consumed by o_proj; TP cannot infer
            # that distinction through the attention kernel.
            self._replace_deepseek_mla_linear_indices( #TODO it false, if we will want to prune v_out
                group, hf_attn.o_proj._original_component, torch.tensor([], device=q_idxs.device),
                False, self.DG
            )
            
            self._replace_deepseek_mla_linear_indices(
                group, hf_attn.kv_a_layernorm._original_component.weight, kv_lora_idxs, True, self.DG
            )
            #TODO for q_a_layernorm also via q_lora_rank... another indices
            return group
        # --------------------------------------------------------------------- #

        elif (bridge_name.split('.')[-2] == 'attn'):
            proj_name = bridge_name[-2:]
            if (proj_name == '.q' and tp_pruning_function == tp.prune_linear_out_channels):
                
                attn_module = self.model_bridge.get_submodule(bridge_name[:-2])
                assert attn_module._original_component.config is not None
                attn_config = attn_module._original_component.config
                rope_denominator = 1
                #treat it is sign of RoPE (so we need to mirror indices). But it is not 100%?
                if (hasattr(self.model_bridge, 'rotary_emb')):
                    rope_denominator = 2
                    if (isinstance(idxs, float)):
                        idxs = self._convert_head_idx_fraction_to_idxs(idxs, 
                                                                attn_config.head_dim,
                                                                rope_denominator, 
                                                                module._original_component.weight.device
                                                                )
                    
                    idxs = TorchPruningGroupBuilder.close_rope_pairs(idxs, attn_config.head_dim)
                        
                elif(isinstance(idxs, float)):
                    idxs = self._convert_head_idx_fraction_to_idxs(idxs, 
                                                                attn_config.head_dim,
                                                                rope_denominator, 
                                                                module._original_component.weight.device
                                                                )
                
                repeating_heads = cast(int, attn_config.num_attention_heads) 
                idxs = TorchPruningGroupBuilder.manually_indices_repeating(
                    repeating_heads,
                    cast(int, attn_config.head_dim),
                    idxs
                )
            else:
                raise NotImplementedError(f"Pruning for {bridge_name} with {tp_pruning_function} is not implemented yet. For attn module q,k,v and o linked. But torch_pruning can find group correctly only for q module.")
        
        else:
            if (isinstance(idxs, float)):
                idxs = self._convert_linear_idx_fraction_to_idxs(idxs, module, tp_pruning_function)
        
        group = self.DG.get_pruning_group(
            module._original_component, 
            tp_pruning_function, 
            idxs=idxs.tolist()
        )
        
        self._fix_group(group)
        return group

    def _fix_group(self, group: tp.Group):
        '''Fix torch pruning group indices for k and v modules in attn (for example for kv_repeat)'''
        for _, (dep, idxs) in enumerate(group): #type: ignore
            if (isinstance(dep.layer, torch.nn.Linear)):
                original_bridge = cast(LinearBridge, self.model_bridge.get_submodule(dep.target.name[:dep.target.name.find(" ") - TRANSFORMER_LENS_ORIGINAL_COMPONENT_SUFFIX_LEN]))
                #I don't find a better way to get universal transformer_lens name, only from name of hook
                assert original_bridge.hook_in.name is not None
                bridge_name = original_bridge.hook_in.name[:-HOOK_IN_NAME_LEN]
                if (bridge_name.endswith(".k") or bridge_name.endswith(".v")):
                    in_channels_need_to_be_pruned = dep.pruning_fn.__name__.endswith("in_channels")
                    if (not in_channels_need_to_be_pruned):
                        attn_module = self.model_bridge.get_submodule(bridge_name[:-2])
                        attn_config = attn_module._original_component.config
                        assert attn_config is not None
                        for j in idxs[::-1]:
                            if (j >= original_bridge.out_features):
                                fixed_indices = TorchPruningGroupBuilder.map_q_indices_to_kv(
                                    q_idxs=idxs,
                                    num_q_heads=attn_config.num_attention_heads,
                                    num_kv_heads=attn_config.num_key_value_heads,
                                    head_dim=attn_config.head_dim
                                )
                                TorchPruningGroupBuilder.replace_linear_indices(group, dep.layer, fixed_indices, True, self.DG)
                                break
                    
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

        mapped_hybrid_idxs = mapper(hybrid_q_idxs) #type: ignore

        # GQA mapping is many-to-one:
        # some Q-heads maps on one KV-head.
        kv_idxs = sorted({
            int(mapped.idx)
            for mapped in mapped_hybrid_idxs
        })

        return kv_idxs

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
