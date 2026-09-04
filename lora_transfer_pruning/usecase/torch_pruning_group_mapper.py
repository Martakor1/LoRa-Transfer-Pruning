import peft
import torch
import torch_pruning as tp
from transformers import DeepseekV2Model
from lora_transfer_pruning.core.pruning_instrumentor import PruningInstrumentor
from transformer_lens.model_bridge.bridge import TransformerBridge
from typing import Callable, cast
from lora_transfer_pruning.core.constants import HOOK_IN_NAME_LEN, TRANSFORMER_LENS_ORIGINAL_COMPONENT_SUFFIX_LEN

from transformer_lens.model_bridge.generalized_components.base import GeneralizedComponent
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge
from transformer_lens.model_bridge.generalized_components.mla_attention import MLAAttentionBridge

class TorchPruningGroupMapper:
    _WEIGHT_SUFFIX_LEN = len(".weight")
    
    @staticmethod
    def prepare_group_for_transfer_pruning(model_bridge: TransformerBridge,
                                           group: tp.Group,
                                           rescale=True,
                                           prune_task_name: str | None = None,
                                           active_adapters_source: Callable[[], list[str]] | None = None):        
        '''Prepare a group of dependencies for transfer pruning by adding hooks to the group's modules. 
        This function modifies the model in place, adding hooks to the layers that need to be pruned.
        
        Provide `prune_task_name` if you use LoRA and need to prepare pruning for a specific pruning task (e.g. only for specific LoRA adapter) (transfer pruning).
        The pruning hooks will be active only when concrete LoRA adapter is active.'''
        device = model_bridge.device
        if (prune_task_name is not None):
            if (active_adapters_source is None):
                raise ValueError("active_adapters_source must be provided when prune_task_name is provided.")

        for i, (dep, idxs) in enumerate(group): # type: ignore
            #don't add activation hooks on internal and backward operations
            if (not type(dep.layer).__module__.startswith("torch_pruning.ops")):
                out_channels_need_to_be_pruned = dep.pruning_fn.__name__.endswith("out_channels")
                idxs = torch.tensor(idxs, device=device)
                
                if (isinstance(dep.layer, torch.nn.Linear)):
                    #no need to add hooks for modules changed in in_channels, 
                    #because activation already was pruned (zeroed) in previous out
                    if (out_channels_need_to_be_pruned or (i == 0)):
                        original_bridge = cast(LinearBridge, model_bridge.get_submodule(dep.target.name.rpartition("._original_component")[0]))
                        
                        cols = None
                        rows = idxs
                        if (not out_channels_need_to_be_pruned):
                            cols = idxs
                            rows = None
                        
                        original_component = original_bridge._original_component
                        if (isinstance(original_component, torch.nn.Linear)):
                            PruningInstrumentor.prepare_linear_for_pruning(original_bridge,
                                                                           cols,
                                                                           rows, 
                                                                           rescale=rescale,
                                                                           active_adapters_source=active_adapters_source,
                                                                           adapter_name=prune_task_name)
                        elif (isinstance(original_component, peft.tuners.lora.layer.Linear)):
                            if (dep.layer is original_component.base_layer): #we only need to add hooks for base layer, to avoid double hooks later for lora_A or lora_B
                                PruningInstrumentor.prepare_linear_for_pruning(original_bridge,
                                                                               cols,
                                                                               rows, 
                                                                               rescale=rescale,
                                                                               active_adapters_source=lambda: original_bridge.active_adapters,
                                                                               adapter_name=prune_task_name)
                        else:
                            raise NotImplementedError(f"Pruning for {original_component.__class__.__name__} is not implemented yet, but it is in dependency group")
        
        if (isinstance(model_bridge.model, DeepseekV2Model) and (rescale == False)):
            for i, (dep, idxs) in enumerate(group): #type: ignore
                if (isinstance(dep.layer, torch.nn.Linear)):
                    #if we compress q_proj in deepseek we should change scaling = self._qk_head_dim ** (-0.5)
                    original_bridge = cast(LinearBridge, model_bridge.get_submodule(dep.target.name.rpartition("._original_component")[0]))
                    assert original_bridge.hook_in.name is not None
                    bridge_name = original_bridge.hook_in.name[:-HOOK_IN_NAME_LEN]
                    if (bridge_name.endswith(".q_proj")
                            and len(idxs) > 0 
                            and dep.pruning_fn.__name__.endswith("out_channels")
                        ):
                        attn_bridge = model_bridge.get_submodule(bridge_name[:-7])
                        assert isinstance(attn_bridge, MLAAttentionBridge)
                        PruningInstrumentor.prepare_mla_attention_bridge_for_pruning(attn_bridge, torch.tensor(idxs))

        
    @staticmethod
    def prepare_all_norms(model_bridge: TransformerBridge):
        '''Prepare all norms by adding hooks that dynamically recognize num 
        of pruned channels and make necessary rescaling.
        
        We can't do it in `prepare_group_for_transfer_pruning` because not all norms have weights and thus are not in dependency graph.'''
        for name, module in model_bridge.named_modules():
            if (isinstance(module, TransformerBridge) 
                or isinstance(module, GeneralizedComponent)
                or not name.endswith("._original_component")
            ):
                continue
            
            class_name = module.__class__.__name__.lower()
            if ("rms" in class_name):
                original_rms_bridge = cast(GeneralizedComponent, model_bridge.get_submodule(name[:-TRANSFORMER_LENS_ORIGINAL_COMPONENT_SUFFIX_LEN]))
                PruningInstrumentor.prepare_rms_norm_for_pruning_dynamically(original_rms_bridge)
            else:
                if ("norm" in name or "norm" in class_name):
                    raise NotImplementedError(f"Pruning for norm {class_name} is not implemented yet, but probably should be.")
                
    
    #solution, if norm can be found in dep graph
    #not all norms have weight, so it's useless  
    # elif(isinstance(dep.layer, torch.nn.Parameter)):
    #     assert group._DG is not None, "DependencyGraph is not set for the group."
    #     full_name = group._DG._param_to_name[dep.layer]
    #     if (full_name.endswith(".weight")):
    #         original_component_name = full_name[:-TorchPruningGroupMapper._WEIGHT_SUFFIX_LEN]
    #         class_name = model_bridge.get_submodule(original_component_name).__class__.__name__.lower()
    #         if ("rms" in class_name):
    #             original_bridge = cast(GeneralizedComponent, model_bridge.get_submodule(original_component_name[:-TRANSFORMER_LENS_ORIGINAL_COMPONENT_SUFFIX_LEN]))
    #             assert out_channels_need_to_be_pruned
    #             PruningInstrumentor.prepare_rms_norm_for_pruning(original_bridge, idxs)
    #         else:
    #             raise NotImplementedError(f"Pruning for {class_name} is not implemented yet, but it is in dependency group")
    #     else:
    #         print(f"Parameter {full_name} is not weight, ignore it for pruning prepairing.")
    