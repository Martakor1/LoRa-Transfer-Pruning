"""Prepare a model through independently callable stages."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from lora_transfer_pruning.port.lora_injector import LoraInjector
from lora_transfer_pruning.port.model_bridger import ModelBridger
from lora_transfer_pruning.port.model_loader import ModelLoader
from lora_transfer_pruning.port.model_quantizer import LayerFilter, ModelQuantizer

if TYPE_CHECKING:
    import torch
    from peft import LoraConfig, PeftModel
    from transformer_lens.model_bridge import TransformerBridge
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


class PrepareModelUseCase:
    """Prepare a pretrained model for LoRA training through a TransformerLens bridge."""

    def __init__(
        self,
        loader: ModelLoader | None = None,
        lora: LoraInjector | None = None,
        quantizer: ModelQuantizer | None = None,
        bridger: ModelBridger | None = None,
    ) -> None:
        """Use the supplied adapters or the defaults for each stage."""
        if loader is None:
            from lora_transfer_pruning.adapter.model_loading.hugging_face_model_loader import HuggingFaceModelLoader

            loader = HuggingFaceModelLoader()
        if lora is None:
            import peft
            from lora_transfer_pruning.adapter.lora_injection.peft_lora_injector import PeftLoraInjector
            from lora_transfer_pruning.adapter.quantization.torchao_quantizer_int4.torchao_int4_linear import TorchAOInt4Linear

            lora = PeftLoraInjector(
                {TorchAOInt4Linear: peft.tuners.lora.layer.Linear}
            )
        if quantizer is None:
            from lora_transfer_pruning.adapter.quantization.torchao_quantizer_int4.torchao_quantizer_int4 import TorchAOQuantizerInt4

            quantizer = TorchAOQuantizerInt4()
        if bridger is None:
            from lora_transfer_pruning.adapter.bridging.transformer_lens_bridge_adapter import TransformerLensBridgeAdapter

            bridger = TransformerLensBridgeAdapter()

        self._loader = loader
        self._lora = lora
        self._quantizer = quantizer
        self._bridger = bridger

    def load_quantize_lora_and_bridge(
        self,
        model_name: str | Path,
        lora_config: LoraConfig,
        *,
        device: torch.device | str,
        dtype: torch.dtype | Literal["auto"] = "auto",
        defuse_experts: bool = True,
    ) -> TransformerBridge:
        """Load -> quantize -> inject LoRA -> and bridge the model in that order.

        Args:
            model_name: Hugging Face model ID or local checkpoint path.
            lora_config: Configuration passed to the LoRA injector.
            device: Device on which to perform quantization.
            dtype: Dtype to use when loading weights. Auto follows the checkpoint.
            defuse_experts: Whether to split fused MoE experts during loading.
        """
        model = self.load_model(model_name, dtype=dtype, defuse_experts=defuse_experts)
        model = self.quantize(model, device=device)
        adapted_model = self.add_lora(model, lora_config)
        return self.bridge_model(adapted_model)

    def load_model(
        self,
        model_name: str | Path,
        *,
        dtype: torch.dtype | Literal["auto"] = "auto",
        device_map: str | dict[str, str | int | torch.device] | torch.device | int | None = None,
        defuse_experts: bool = True,
        **options: Any,
    ) -> PreTrainedModel:
        """Load the checkpoint and optionally split fused MoE expert weights."""
        return self._loader.load(
            model_name, dtype=dtype, device_map=device_map,
            defuse_experts=defuse_experts, **options,
        )

    def add_lora(
        self,
        model: PreTrainedModel,
        config: LoraConfig,
        *,
        adapter_name: str = "default",
        autocast_adapter_dtype: bool = True,
    ) -> PeftModel:
        """Attach a named LoRA adapter using the supplied PEFT configuration."""
        return self._lora.add_lora(
            model, config, adapter_name=adapter_name,
            autocast_adapter_dtype=autocast_adapter_dtype,
        )

    def quantize(
        self,
        model: PreTrainedModel,
        *,
        device: torch.device | str | None = None,
        filter_fn: LayerFilter | None = None,
    ) -> PreTrainedModel:
        """Quantize eligible layers on device, or on their current devices."""
        return self._quantizer.quantize(model, device=device, filter_fn=filter_fn)

    def bridge_model(
        self,
        model: PreTrainedModel | PeftModel | TransformerBridge,
        *,
        tokenizer: PreTrainedTokenizerBase | None = None,
        model_name: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> TransformerBridge:
        """Wrap the prepared model with TransformerLens or reuse an existing bridge."""
        return self._bridger.bridge(
            model, tokenizer=tokenizer, model_name=model_name, dtype=dtype,
        )
