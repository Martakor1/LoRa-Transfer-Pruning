import torch
import torch_pruning as tp
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from torch import nn
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge
from transformers import PreTrainedConfig
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2Model
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention
from lora_transfer_pruning.core import tl_utils
from lora_transfer_pruning.core.prune_task_type import GroupPruneTask
from lora_transfer_pruning.adapter.quantization.torchao_quantizer_int4.torchao_int4_linear import TorchAOInt4Linear

from typing import Callable, cast

from .torch_pruning_group_task import TorchPruningGroupTask
from .universal.universal_group_builder import UniversalGroupBuilder
from .deepseek.deepseek_v2_group_builder import DeepseekV2GroupBuilder
from .index_utils import IndexUtils
from .tp_utils import get_active_module_in_dep_graph_from_linear_bridge
from . import tp_utils
from .universal.gemma4_group_builder import Gemma4GroupBuilder
from .custom_pruners import PackedLinearDepGraphOnlyPruner
from lora_transfer_pruning.core.tl_adapters import ExpertBridge
from torch_pruning.pruner.function import BasePruningFunc

class TorchPruningGroupBuilder:

    def __init__(self,
                 model_bridge: TransformerBridge,
                 example_input_ids: torch.Tensor,
                 unwrapped_params: list[torch.nn.Parameter] = [],
                 ignored_params: list[torch.nn.Parameter] = [],
                 customized_pruners: dict | None = None):
        '''Builds a dependency graph for pruning using torch_pruning.

            Params:
                model_bridge: An instance of TransformerBridge that wraps the model to be pruned.
                example_input_ids: A tensor representing batches of input IDs for the model (need for torch_pruning to build dep graph by tracing the forward pass).
                unwrapped_params: params in modules, that torch_pruning can't handle or trailing params that not in modules. For example .weight in `LlamaRMSNorm`.
                    Pass it to ignore them during pruning.
        '''
        pruners: dict[type[nn.Module], BasePruningFunc] = {
            TorchAOInt4Linear: PackedLinearDepGraphOnlyPruner(),
            TorchLinear: PackedLinearDepGraphOnlyPruner(),
        }
        pruners.update(customized_pruners or {})

        self.DG = tp.DependencyGraph().build_dependency(
            model_bridge,
            example_inputs=example_input_ids,
            forward_fn=TorchPruningGroupBuilder._trace_forward, #type: ignore 
            ignored_params=ignored_params,
            customized_pruners=pruners,
            unwrapped_parameters=list(
                zip(unwrapped_params, [0] * len(unwrapped_params)))  # type: ignore
        )
        self.model_bridge = model_bridge

    @staticmethod
    def _trace_forward(model_bridge: TransformerBridge, input_ids) -> tuple[torch.Tensor, ...]:
        model_output = model_bridge(
                input=input_ids,
                use_cache=False,
                return_type="logits",
            )
        
        experts_outputs = TorchPruningGroupBuilder._trace_all_experts(model_bridge)

        return (model_output, *experts_outputs)
    
    @staticmethod
    def _trace_all_experts(model_bridge: TransformerBridge) -> list[torch.Tensor]:
        with torch.autograd.graph.save_on_cpu(pin_memory=False): #todo benefits of pin?
            expert_list_modules = [
                module for name, module in model_bridge.named_modules()
                if name.rsplit('.', 1)[-1] == 'experts' and isinstance(module, nn.ModuleList)
            ]
            experts = cast(list[ExpertBridge], [expert for expert_list in expert_list_modules for expert in expert_list])
            if not experts:
                return []
            
            extra_outputs = []
            synthetic_input = torch.ones(
                [1, model_bridge.config.hidden_size], device=model_bridge.device, dtype=model_bridge.dtype,
                requires_grad=True,
            )
            for expert in experts:
                print(f"Tracing expert {tl_utils.get_absolute_tl_name(expert)} with synthetic input")

                extra_outputs.append(expert(synthetic_input))

            return extra_outputs


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
        bridge_name = tl_utils.get_absolute_tl_name(module)

        device = tp_utils.get_device(module)
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
        try:
            self.model_bridge.original_model.requires_grad_(True) #we use whole model here, because dependencies can goes far away from supplied module
            
            if (".attn" in bridge_name):
                # --------------------------------------------------------------------- #
                # DeepSeek V2 MLA direct-Q path.
                # --------------------------------------------------------------------- #
                if (isinstance(self.model_bridge.model, DeepseekV2Model)
                    and bridge_name.endswith(".attn.q_proj")
                    and tp_pruning_function == tp.prune_linear_out_channels
                        ):
    
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
                    return group, structural_setup
                    
                # --------------------------------------------------------------------- #
                
                elif (bridge_name.endswith(".attn.q") and tp_pruning_function == tp.prune_linear_out_channels):
                    hf_attn = self.model_bridge.get_submodule(bridge_name.rsplit('.', 1)[0])._original_component
                    if isinstance(hf_attn, Gemma4TextAttention):
                        return Gemma4GroupBuilder.get_correct_pruning_group_and_structural_setup_for_attn(
                            self.DG, self.model_bridge, tp_pruning_function,
                            bridge_name, idxs, device)
    
                    idxs, setup_function = UniversalGroupBuilder.get_correct_pruning_group_and_structural_setup_for_attn(
                        self.model_bridge,
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
            
            return group, setup_function
        finally:
            TorchPruningGroupBuilder.load_grad_mask(grad_mask)

    def _fix_group(self, group: tp.Group):
        '''Fix torch pruning group indices for k and v modules in attn (for example for kv_repeat)'''
        for _, (dep, idxs) in enumerate(group):  # type: ignore
            if (isinstance(dep.layer, torch.nn.Linear)):
                original_bridge = cast(LinearBridge, self.model_bridge.get_submodule(
                    dep.target.name.rpartition("._original_component")[0]))
                # I don't find a better way to get universal transformer_lens name, only from name of hook
                bridge_name = tl_utils.get_absolute_tl_name(original_bridge)
                if (bridge_name.endswith(".k") or bridge_name.endswith(".v")):
                    in_channels_need_to_be_pruned = dep.pruning_fn.__name__.endswith("in_channels")
                    # for in channels can't be in invalid state
                    if (not in_channels_need_to_be_pruned):
                        attn_module = self.model_bridge.get_submodule(
                            bridge_name[:-2])
                        
                        attn_config = cast(PreTrainedConfig, getattr(attn_module._original_component, "config", None))
                        assert attn_config is not None
                        if any(j >= original_bridge.out_features for j in idxs):
                            fixed_indices = IndexUtils.map_q_indices_to_kv(
                                q_idxs=idxs,
                                num_q_heads=attn_config.num_attention_heads,
                                num_kv_heads=attn_config.num_key_value_heads,
                                head_dim=attn_module._original_component.head_dim, #type: ignore
                            )
                        elif len(set(idxs)) != len(idxs):
                            # TP's GQA mapping (_GQAIndexMapping in tp.) can already be in KV coordinates,
                            # but with one copy per repeated query head. The Linear
                            # pruner subtracts len(idxs) from out_features, whereas
                            # weight slicing removes each coordinate only once.
                            fixed_indices = sorted(set(idxs))
                        else:
                            continue
                        UniversalGroupBuilder.replace_linear_indices(
                            group, original_bridge, fixed_indices, True, self.DG)
