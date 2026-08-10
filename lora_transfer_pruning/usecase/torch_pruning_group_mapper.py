import torch
import torch_pruning as tp
from lora_transfer_pruning.core.pruning_instrumentor import PruningInstrumentor
from transformer_lens.model_bridge.bridge import TransformerBridge
from typing import cast
from lora_transfer_pruning.core.constants import TRANSFORMER_LENS_ORIGINAL_COMPONENT_SUFFIX_LEN

from transformer_lens.model_bridge.generalized_components.linear import LinearBridge

class TorchPruningGroupMapper:
    
    @staticmethod
    def prepare_group_for_transfer_pruning(model_bridge: TransformerBridge, group: tp.Group, rescale=True):        
        device = model_bridge.device
        for i, (dep, idxs) in enumerate(group): # type: ignore
            #dont add activation hooks on internal and backward operations
            if (not type(dep.layer).__module__.startswith("torch_pruning.ops")):
                if (not isinstance(dep.layer, torch.nn.Linear)):
                    raise NotImplementedError("Only Linear layers are supported for transfer pruning now.")
                #no need to add hooks for modules changed in in_channels, 
                #because activation already was pruned (zeroed) in previous out
                in_channels_need_to_be_pruned = dep.pruning_fn.__name__.endswith("in_channels")
                if ((not in_channels_need_to_be_pruned) or (i == 0)):
                    original_bridge = cast(LinearBridge, model_bridge.get_submodule(dep.target.name[:dep.target.name.find(" ") - TRANSFORMER_LENS_ORIGINAL_COMPONENT_SUFFIX_LEN]))
                    idxs = torch.tensor(idxs, device=device)
                    if (in_channels_need_to_be_pruned):
                        PruningInstrumentor.prepare_linear_for_pruning(original_bridge, idxs, None, rescale=rescale)
                    else:
                        PruningInstrumentor.prepare_linear_for_pruning(original_bridge, None, idxs, rescale=rescale)


        #if attn is PositionEmbeddingsAttentionBridge => RoPE (weak sign, because there is no 100% guarantee of RoPE)
        #more guarantee - in model.rotary_emb = LlamaRotaryEmbedding() module presence
        #but for activation pruning we dont need to prune cos and sin
        