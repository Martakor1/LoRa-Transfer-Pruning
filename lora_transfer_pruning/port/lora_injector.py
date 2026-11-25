from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from peft import LoraConfig, PeftModel
    from transformers import PreTrainedModel


class LoraInjector(Protocol):
    """Attach LoRA adapters to a pretrained base model."""

    def add_lora(
        self,
        model: PreTrainedModel,
        config: LoraConfig,
        *,
        adapter_name: str,
        autocast_adapter_dtype: bool,
    ) -> PeftModel:
        """Attach a LoRA adapter and return the resulting PEFT model.

        Args:
            model: Pretrained base model, which may already be quantized.
            config: LoRA settings, including rank and target modules.
            adapter_name: Name used to select or manage the adapter later.
            autocast_adapter_dtype: Whether to promote FP16 or BF16 adapter
                weights to FP32 for training stability where supported.
        """
        ...
