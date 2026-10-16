"""Gemma4 head-coordinate pruning, including shared-KV consumers.

Autograd's dependency graph does not encode the meaning of the shared KV dict.
Use its module/parameter nodes, but specify the architectural dependencies here.
"""

from typing import Callable, cast

import torch
from torch import nn
import torch_pruning as tp

from transformer_lens.model_bridge.generalized_components.linear import LinearBridge

from .universal_structural_setup import UniversalStructuralSetup
from .universal_group_builder import UniversalGroupBuilder
from ..index_utils import IndexUtils
from ..tp_utils import get_active_module_in_dep_graph_from_linear_bridge
from transformer_lens.model_bridge.bridge import TransformerBridge


class Gemma4GroupBuilder:
    
    @staticmethod
    def shared_kv_layers_family(model_bridge, layer_idx):
        attn = model_bridge.blocks[layer_idx].attn._original_component
        if not (attn.is_kv_shared_layer or attn.store_full_length_kv):
            return [layer_idx]
        return [
            i for i, block in enumerate(model_bridge.blocks)
            if block.attn._original_component.layer_type == attn.layer_type
            and (block.attn._original_component.is_kv_shared_layer
                 or block.attn._original_component.store_full_length_kv)
        ]

    @staticmethod
    def get_correct_pruning_group_and_structural_setup_for_attn(
        DG: tp.DependencyGraph,
        model_bridge: TransformerBridge,
        tp_pruning_function: Callable,
        bridge_name: str,
        idxs: torch.Tensor | float,
        device: torch.device,
    ) -> tuple[tp.Group, Callable[[], None]]:
        '''Use ordinary head-index/setup helpers, with explicit shared-KV targets.

        A traced attention graph alone does not reliably express cross-layer KV
        dependencies. Build their module boundaries explicitly from DG nodes.
        '''
        layer_idx = int(bridge_name.split('.')[1])
        layers_family = Gemma4GroupBuilder.shared_kv_layers_family(model_bridge, layer_idx)
        root_attn = model_bridge.blocks[layer_idx].attn._original_component
        head_dim = root_attn.head_dim  # full attention may differ from config.head_dim!
        local_idxs = UniversalGroupBuilder.prepare_local_attn_head_indices(
            idxs, head_dim, True, device)

        group = tp.Group()
        group._DG = DG #type: ignore
        root_bridge = cast(LinearBridge, model_bridge.get_submodule(bridge_name))
        if not isinstance(root_bridge._original_component, nn.Linear):
            raise NotImplementedError('Gemma shared-KV structural groups currently require plain Linear, not LoRA')
        root = get_active_module_in_dep_graph_from_linear_bridge(root_bridge, tp_pruning_function)
        source = DG.module2node[root]
        seen = set()

        def add_in_dep_graph(module, handler, indices):
            if module is None or (id(module), handler) in seen:
                return
            if module not in DG.module2node:
                raise ValueError(f'Gemma pruning target is missing from DG: {type(module).__name__}')
            seen.add((id(module), handler))
            group.add_dep(tp.Dependency(tp_pruning_function, handler,
                                        source, DG.module2node[module]), indices.tolist())

        def add_linear_in_dep_graph(bridge, outgoing):
            module = bridge._original_component
            if not isinstance(module, nn.Linear):
                raise NotImplementedError('Gemma structural groups require plain Linear targets')
            size = module.out_features if outgoing else module.in_features
            if size % head_dim:
                raise ValueError('Gemma projection width is not divisible by its actual head_dim')
            indices = IndexUtils.manually_indices_repeating(size // head_dim, head_dim, local_idxs)
            add_in_dep_graph(module, tp.prune_linear_out_channels if outgoing else tp.prune_linear_in_channels, indices)

        # Root must be first: Group.prune uses it for recording history.
        add_linear_in_dep_graph(model_bridge.blocks[layer_idx].attn.q, True)
        setups = []
        for i in layers_family:
            bridge = model_bridge.blocks[i].attn
            hf_attn = bridge._original_component
            if hf_attn.head_dim != head_dim:
                raise ValueError('Shared KV family has inconsistent head dimensions')
            add_linear_in_dep_graph(bridge.q, True)
            add_linear_in_dep_graph(bridge.o, False)
            for norm_name in ('q_norm', 'k_norm'):
                norm = getattr(hf_attn, norm_name, None)
                norm = getattr(norm, '_original_component', norm)
                weight = getattr(norm, 'weight', None)
                if weight is not None:
                    add_in_dep_graph(weight, tp.prune_parameter_out_channels, local_idxs)
            if not hf_attn.is_kv_shared_layer:
                add_linear_in_dep_graph(bridge.k, True)
                if hf_attn.v_proj is not None:
                    add_linear_in_dep_graph(bridge.v, True)
            setups.append(UniversalStructuralSetup.get_setup_function_for_ordinary_attn(
                hf_attn, bridge, local_idxs))

        def setup():
            for apply in setups:
                apply()

        # LocalPruning uses this to prevent applying the same shared family twice.
        group.gemma_kv_family = tuple(layers_family) #TODO вынести это в контракт prune_task по аналогии с kv_lora_idxs_deepseek, чтобы ставить индексы на всю family indices_by_family
        group.gemma_local_indices = tuple(local_idxs.tolist())
        return group, setup
