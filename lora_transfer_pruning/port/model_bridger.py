from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch
    from peft import PeftModel
    from transformer_lens.model_bridge import TransformerBridge
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


class ModelBridger(Protocol):
    """Expose a loaded model through a TransformerLens bridge."""

    def bridge(
        self,
        model: PreTrainedModel | PeftModel | TransformerBridge,
        *,
        tokenizer: PreTrainedTokenizerBase | None,
        model_name: str | None,
        dtype: torch.dtype | None,
    ) -> TransformerBridge:
        """Wrap a loaded model or return an existing bridge unchanged.

        Args:
            model: Loaded base model, PEFT model, or existing bridge.
            tokenizer: Tokenizer to use. None delegates selection to the bridge.
            model_name: Model identifier used to initialize the bridge. None uses
                the loaded model's identifier when available.
            dtype: Dtype for the bridge configuration. None uses the loaded
                model's dtype.
        """
        ...
