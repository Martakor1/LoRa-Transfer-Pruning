from __future__ import annotations

from typing import TYPE_CHECKING, cast

from peft import PeftModel
from transformer_lens.model_bridge import TransformerBridge
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from lora_transfer_pruning.port.model_bridger import ModelBridger

if TYPE_CHECKING:
    import torch


class TransformerLensBridgeAdapter(ModelBridger):
    
    def bridge(
        self,
        model: PreTrainedModel | PeftModel | TransformerBridge,
        *,
        tokenizer: PreTrainedTokenizerBase | None = None,
        model_name: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> TransformerBridge:
        if isinstance(model, TransformerBridge):
            return model
        base = cast(PreTrainedModel, model.get_base_model()) if isinstance(model, PeftModel) else model
        trainable = {p for p in model.parameters() if p.requires_grad}

        # TransformerLens selects the registered architecture and configures attention.
        # Match cfg.dtype to the loaded weights and preserve their device placement.
        bridge = TransformerBridge.boot_transformers(
            model_name or base.name_or_path or "external",
            hf_model=base,
            tokenizer=tokenizer,
            dtype=base.dtype if dtype is None else dtype,
            device=None if getattr(base, "hf_device_map", None) else base.device,
        )
        parameters = set(model.parameters())
        if not trainable <= parameters:
            raise RuntimeError("TransformerLens replaced trainable model parameters")
        
        # note: An unembedding bridge can add a trainable zero bias to a bias-free head.
        return bridge
