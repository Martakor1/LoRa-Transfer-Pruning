from transformer_lens.model_bridge.generalized_components.base import GeneralizedComponent


TRANSFORMER_LENS_ORIGINAL_COMPONENT_SUFFIX_LEN = len("._original_component")
HOOK_IN_NAME_LEN = len(".hook_in")

def get_absolute_tl_name(module: GeneralizedComponent) -> str:
    #bridge.blocks[0].mlp.experts[0].hook_in.name -> 'blocks.0.mlp.experts.0.hook_in'
    #bridge.blocks[0].mlp.experts[0].name -> 'experts.0'
    assert module.hook_in.name is not None
    return module.hook_in.name[:-HOOK_IN_NAME_LEN]
