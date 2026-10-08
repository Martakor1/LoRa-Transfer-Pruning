import peft
import torch
import torch_pruning as tp
from torch import nn
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge
from transformers import PreTrainedConfig
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2Model
from lora_transfer_pruning.core.constants import HOOK_IN_NAME_LEN
from lora_transfer_pruning.core.prune_task_type import GroupPruneTask

from typing import Callable, cast

from .torch_pruning_group_task import TorchPruningGroupTask
from .universal.universal_group_builder import UniversalGroupBuilder
from .deepseek.deepseek_v2_group_builder import DeepseekV2GroupBuilder
from .index_utils import IndexUtils
from .tp_utils import get_active_module_in_dep_graph_from_linear_bridge

class TorchPruningGroupBuilder:

    def __init__(self,
                 model_bridge: TransformerBridge,
                 example_input_ids: torch.Tensor,
                 unwrapped_params: list[torch.nn.Parameter] = [],
                 ignored_params: list[torch.nn.Parameter] = []):
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
            ignored_params=ignored_params,
            unwrapped_parameters=list(
                zip(unwrapped_params, [0] * len(unwrapped_params)))  # type: ignore
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
    def get_device(module: LinearBridge) -> torch.device:
        '''Get device of the module's original component.'''
        if isinstance(module._original_component, nn.Linear):
            return module._original_component.weight.device
        elif isinstance(module._original_component, peft.tuners.lora.layer.Linear):
            return cast(torch.device, module._original_component.base_layer.weight.device)
        else:
            raise TypeError(
                f"Unsupported module type {type(module._original_component)} for getting device.")
    

    @staticmethod
    def save_grad_mask(model: nn.Module) -> dict[nn.Parameter, bool]:
        mask = {}
        for p in model.parameters():
            mask[p] = p.requires_grad
        return mask
    
    @staticmethod
    def load_grad_mask(mask: dict[nn.Parameter, bool]):
        for p, saved_requires_grad in mask.items():
            p.requires_grad = saved_requires_grad

    def get_correct_pruning_group_and_structural_setup(self, module: LinearBridge, tp_pruning_function: Callable, groupPruneTask: GroupPruneTask) -> tuple[tp.Group, Callable[[], None]]:
        '''Creates torch pruning group and fixes indices for k and v modules in attn (for example for kv_repeat).

        For rope mirrors indices.

        Also returns setup function to apply later. It change static fields related to module for torch pruning shapes compatability (self_attn.head_dim for example).
        If nothing to change returns no-op function.'''
        assert module.hook_in.name is not None
        bridge_name = module.hook_in.name[:-HOOK_IN_NAME_LEN]

        device = TorchPruningGroupBuilder.get_device(module)
        tp_group_task = TorchPruningGroupTask.from_group_prune_task(groupPruneTask, device)

        idxs = tp_group_task.rows
        if (tp_pruning_function == tp.prune_linear_in_channels):
            idxs = tp_group_task.cols

        assert idxs is not None

        def setup_function(): return None  # Default no-op setup function
        
        # Needs in case of LoRA to register .base_layer(requires_grad=True) with adapters in dependency graph 
        # for correct future torch pruning reshape of .base_layer module. Otherwise there
        # will be incompatibility of shapes between .base_layer and .lora_A/.lora_B modules after pruning.
        grad_mask = TorchPruningGroupBuilder.save_grad_mask(self.model_bridge)
        self.model_bridge.original_model.requires_grad_(True) #we use whole model here, because dependencies can goes far away from supplied module
        
        if (".attn" in bridge_name):
            # --------------------------------------------------------------------- #
            # DeepSeek V2 MLA direct-Q path.
            # --------------------------------------------------------------------- #
            if (bridge_name.endswith(".attn.q_proj")
                and tp_pruning_function == tp.prune_linear_out_channels
                    and isinstance(self.model_bridge.model, DeepseekV2Model)):

                group, structural_setup = DeepseekV2GroupBuilder.get_correct_pruning_group_and_structural_setup_for_attn(
                    self.DG, 
                    self.model_bridge, 
                    module, 
                    tp_group_task, 
                    tp_pruning_function,
                    bridge_name,
                    idxs,
                    device
                )
                TorchPruningGroupBuilder.load_grad_mask(grad_mask)
                return group, structural_setup
                
            # --------------------------------------------------------------------- #

            elif (bridge_name.split('.')[-2] == 'attn'):
                idxs, setup_function = UniversalGroupBuilder.get_correct_pruning_group_and_structural_setup_for_attn(
                    self.model_bridge,
                    tp_pruning_function,
                    bridge_name,
                    idxs,
                    device)
            else:
                raise NotImplementedError(
                    f"Pruning for {bridge_name} with {tp_pruning_function} is not implemented yet. Wrong module to create pruning group in attn or unsupported pruning function.")
        else:
            if (isinstance(idxs, float | int)):
                idxs = IndexUtils.convert_linear_idx_fraction_to_idxs(
                    idxs, module, tp_pruning_function)

        group = self.DG.get_pruning_group(
            get_active_module_in_dep_graph_from_linear_bridge(
                module, tp_pruning_function),
            tp_pruning_function,
            idxs=idxs.tolist()
        )

        self._fix_group(group)
        
        TorchPruningGroupBuilder.load_grad_mask(grad_mask)
        return group, setup_function

    def _fix_group(self, group: tp.Group):
        '''Fix torch pruning group indices for k and v modules in attn (for example for kv_repeat)'''
        for _, (dep, idxs) in enumerate(group):  # type: ignore
            if (isinstance(dep.layer, torch.nn.Linear)):
                original_bridge = cast(LinearBridge, self.model_bridge.get_submodule(
                    dep.target.name.rpartition("._original_component")[0]))
                # I don't find a better way to get universal transformer_lens name, only from name of hook
                assert original_bridge.hook_in.name is not None
                bridge_name = original_bridge.hook_in.name[:-HOOK_IN_NAME_LEN]
                if (bridge_name.endswith(".k") or bridge_name.endswith(".v")):
                    in_channels_need_to_be_pruned = dep.pruning_fn.__name__.endswith("in_channels")
                    # for in channels can't be in invalid state
                    if (not in_channels_need_to_be_pruned):
                        attn_module = self.model_bridge.get_submodule(
                            bridge_name[:-2])
                        
                        attn_config = cast(PreTrainedConfig, getattr(attn_module._original_component, "config", None))
                        assert attn_config is not None
                        for j in idxs[::-1]:
                            if (j >= original_bridge.out_features):
                                fixed_indices = IndexUtils.map_q_indices_to_kv(
                                    q_idxs=idxs,
                                    num_q_heads=attn_config.num_attention_heads,
                                    num_kv_heads=attn_config.num_key_value_heads,
                                    head_dim=attn_config.head_dim
                                )
                                UniversalGroupBuilder.replace_linear_indices(
                                    group, original_bridge, fixed_indices, True, self.DG)
                                break