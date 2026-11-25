from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.generalized_components.base import GeneralizedComponent
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge
from transformer_lens.model_bridge.supported_architectures.qwen3_moe import Qwen3MoeArchitectureAdapter


class ExpertBridge(GeneralizedComponent):
    is_list_item = True


class Qwen3MoeWithExpertBridges(Qwen3MoeArchitectureAdapter):
    """Expose separate expert projections in Qwen3 MoE models that already have them."""

    def __init__(self, cfg: TransformerBridgeConfig) -> None:
        super().__init__(cfg)
        mlp = self.component_mapping["blocks"].submodules["mlp"] #type: ignore
        mlp.submodules["experts"] = ExpertBridge(
            name="experts",
            submodules={
                "gate_proj": LinearBridge(name="gate_proj"),
                "up_proj": LinearBridge(name="up_proj"),
                "down_proj": LinearBridge(name="down_proj"),
            },
        )