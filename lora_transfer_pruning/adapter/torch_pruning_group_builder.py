from typing import Callable, cast

import torch

import transformer_lens
import transformer_lens.model_bridge
from transformer_lens.model_bridge.bridge import TransformerBridge
import torch_pruning as tp

import transformer_lens.model_bridge.generalized_components
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge
from lora_transfer_pruning.core.constants import TRANSFORMER_LENS_ORIGINAL_COMPONENT_SUFFIX_LEN

class TorchPruningGroupBuilder:
    _HOOK_NAME_LEN = len(".hook_in")    

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
    def close_rope_pairs(local_idxs: torch.Tensor, head_dim: int):
        half = head_dim // 2
        paired = torch.where(
            local_idxs < half,
            local_idxs + half,
            local_idxs - half,
        )
        return torch.unique(torch.cat([local_idxs, paired]), sorted=True)
    
    @staticmethod
    def manually_indices_repeating(num_heads: int, head_dim: int, pruning_indices: torch.Tensor):
        all_indices = []
        for head_num in range(num_heads):
            all_indices.append(
                pruning_indices + head_num * head_dim)
        return torch.cat(all_indices)
        
    def get_correct_pruning_group(self, module: LinearBridge, tp_pruning_function: Callable, idxs: torch.Tensor) -> tp.Group:
        '''Creates torch pruning group and fixes indices for k and v modules in attn (for example for kv_repeat).
        
        For rope mirrors indices.'''
        assert module.hook_in.name is not None
        bridge_name = module.hook_in.name[:-TorchPruningGroupBuilder._HOOK_NAME_LEN]
        if (bridge_name.split('.')[-2] == 'attn'):
            proj_name = bridge_name[-2:]
            if (proj_name == '.q' and tp_pruning_function == tp.prune_linear_out_channels):
                
                #treat it is sign of RoPE (so we need to mirror indices). But it is not 100%. TODO: check
                attn_module = self.model_bridge.get_submodule(bridge_name[:-2])
                if (isinstance(attn_module, transformer_lens.model_bridge.generalized_components.PositionEmbeddingsAttentionBridge)):
                    idxs = TorchPruningGroupBuilder.close_rope_pairs(idxs,attn_module.config.head_dim)
                
                repeating_heads = cast(int, attn_module.config.n_heads) 
                idxs = TorchPruningGroupBuilder.manually_indices_repeating(
                    repeating_heads,
                    cast(int, attn_module.config.head_dim),
                    idxs
                )
            else:
                raise NotImplementedError(f"Pruning for {bridge_name} with {tp_pruning_function} is not implemented yet. For attn module q,k,v and o linked. But torch_pruning can find group correctly only for q module.")
        
        group = self.DG.get_pruning_group(
            module._original_component, 
            tp_pruning_function, 
            idxs=idxs.tolist()
        )
        
        self._fix_group(group)
        return group

    def _fix_group(self, group: tp.Group):
        '''Fix torch pruning group indices for k and v modules in attn (for example for kv_repeat)'''
        for i, (dep, idxs) in enumerate(group): #type: ignore
            if (isinstance(dep.layer, torch.nn.Linear)):
                original_bridge = cast(LinearBridge, self.model_bridge.get_submodule(dep.target.name[:dep.target.name.find(" ") - TRANSFORMER_LENS_ORIGINAL_COMPONENT_SUFFIX_LEN]))
                #I dont find a better way to get universal transformer_lens name, only from name of hook
                assert original_bridge.hook_in.name is not None
                bridge_name = original_bridge.hook_in.name[:-TorchPruningGroupBuilder._HOOK_NAME_LEN]
                if (bridge_name.endswith(".k") or bridge_name.endswith(".v")):
                    in_channels_need_to_be_pruned = dep.pruning_fn.__name__.endswith("in_channels")
                    if (not in_channels_need_to_be_pruned):
                        attn_module = self.model_bridge.get_submodule(bridge_name[:-2])
                        for j in idxs[::-1]:
                            if (j >= original_bridge.out_features):
                                fixed_indices = TorchPruningGroupBuilder._map_q_indices_to_kv(
                                    q_idxs=idxs,
                                    num_q_heads=attn_module.config.n_heads,
                                    num_kv_heads=attn_module.config.n_key_value_heads,
                                    head_dim=attn_module.config.head_dim
                                )
                                TorchPruningGroupBuilder.replace_linear_indices(group, dep.layer, fixed_indices, True, self.DG)
                                break
                    
    @staticmethod
    def _map_q_indices_to_kv(
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
