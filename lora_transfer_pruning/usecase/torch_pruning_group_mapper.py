import torch
import torch_pruning as tp
from transformers import DeepseekV2Model
from lora_transfer_pruning.core.pruning_instrumentor import PruningInstrumentor
from transformer_lens.model_bridge.bridge import TransformerBridge
from typing import cast
from lora_transfer_pruning.core.constants import HOOK_IN_NAME_LEN, TRANSFORMER_LENS_ORIGINAL_COMPONENT_SUFFIX_LEN

from transformer_lens.model_bridge.generalized_components.base import GeneralizedComponent
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge
from transformer_lens.model_bridge.generalized_components.mla_attention import MLAAttentionBridge

class TorchPruningGroupMapper:
    _WEIGHT_SUFFIX_LEN = len(".weight")
    
    @staticmethod
    def prepare_group_for_transfer_pruning(model_bridge: TransformerBridge, group: tp.Group, rescale=True):        
        device = model_bridge.device
        for i, (dep, idxs) in enumerate(group): # type: ignore
            #don't add activation hooks on internal and backward operations
            if (not type(dep.layer).__module__.startswith("torch_pruning.ops")):
                out_channels_need_to_be_pruned = dep.pruning_fn.__name__.endswith("out_channels")
                idxs = torch.tensor(idxs, device=device)
                
                if (isinstance(dep.layer, torch.nn.Linear)):
                    #no need to add hooks for modules changed in in_channels, 
                    #because activation already was pruned (zeroed) in previous out
                    if (out_channels_need_to_be_pruned or (i == 0)):
                        original_bridge = cast(LinearBridge, model_bridge.get_submodule(dep.target.name[:dep.target.name.find(" ") - TRANSFORMER_LENS_ORIGINAL_COMPONENT_SUFFIX_LEN]))
                        if (out_channels_need_to_be_pruned):
                            PruningInstrumentor.prepare_linear_for_pruning(original_bridge, None, idxs, rescale=rescale)
                        else:
                            PruningInstrumentor.prepare_linear_for_pruning(original_bridge, idxs, None, rescale=rescale)
        
        if (isinstance(model_bridge.model, DeepseekV2Model) and (rescale == False)):
            for i, (dep, idxs) in enumerate(group): #type: ignore
                if (isinstance(dep.layer, torch.nn.Linear)):
                    #if we compress q_proj in deepseek we should change scaling = self._qk_head_dim ** (-0.5)
                    original_bridge = cast(LinearBridge, model_bridge.get_submodule(dep.target.name[:dep.target.name.find(" ") - TRANSFORMER_LENS_ORIGINAL_COMPONENT_SUFFIX_LEN]))
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
        of pruned channels and make necessary rescaling.'''
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
    