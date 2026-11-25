from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel

from lora_transfer_pruning.port.model_loader import ModelLoader

class HuggingFaceModelLoader(ModelLoader):
    def __init__(self) -> None:
        pass
    
    def load(
        self,
        model_name: str | Path,
        *,
        dtype: torch.dtype | Literal["auto"] = "auto",
        device_map: str | dict[str, str | int | torch.device] | torch.device | int | None = None,
        defuse_experts: bool = True,
        **options: Any,
    ) -> PreTrainedModel:
        """Load the checkpoint and optionally let Defuser prepare dense experts."""
        # Keep the checkpoint dtype by default. The loader owns device placement.
        model = AutoModelForCausalLM.from_pretrained(
            str(model_name), dtype=dtype, device_map=device_map, **options,
        )
        # Quantized backends such as GPTQ own their expert layout.
        if defuse_experts and not (
            getattr(model, "is_quantized", False)
            or getattr(model, "hf_quantizer", None) is not None
            or getattr(model.config, "quantization_config", None)
        ):
            self._defuse(model)
        return model

    @staticmethod
    def _defuse(model: PreTrainedModel) -> None:
        try:
            from defuser import convert_model
        except ModuleNotFoundError as error:
            # Defuser is optional for environments that load ordinary dense models.
            if error.name != "defuser":
                raise
            return

        convert_model(model, cleanup_original=True)
