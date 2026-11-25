from pathlib import Path
from typing import Any, Literal

import torch
from transformers import PreTrainedModel
from transformers.integrations.accelerate import check_and_set_device_map

from lora_transfer_pruning.port.model_loader import ModelLoader


class GPTQModelLoader(ModelLoader):
    """Load a GPTQ checkpoint with the trainable Torch backend."""

    def __init__(self, device: torch.device | str) -> None:
        self.device = device

    def load(
        self,
        model_name: str | Path,
        *,
        dtype: torch.dtype | Literal["auto"] = "auto",
        device_map: str | dict[str, str | int | torch.device] | torch.device | int | None = None,
        defuse_experts: bool = True,
        **options: Any,
    ) -> PreTrainedModel:
        """Load packed weights. GPTQModel prepares the expert layout during loading."""
        from gptqmodel import GPTQModel
        from gptqmodel.utils.backend import BACKEND

        if not defuse_experts:
            raise ValueError("GPTQModel prepares the expert fusing required by the checkpoint, we can't handle it with external parameter. Set defuse_expert = True")
        if dtype != "auto":
            options["torch_dtype"] = dtype
        gptq_device_map = check_and_set_device_map(device_map) if device_map is not None else None
        if isinstance(gptq_device_map, dict):
            gptq_device_map = {
                path: str(target) if isinstance(target, torch.device) else target
                for path, target in gptq_device_map.items()
            }

        loaded = GPTQModel.load(
            str(model_name),
            device=self.device if device_map is None else None,
            device_map=gptq_device_map,
            backend=BACKEND.GPTQ_TORCH,
            profile="low_memory",
            **options,
        )
        model = loaded.model
        if not isinstance(model, PreTrainedModel):
            raise TypeError(f"Expected a Transformers (PreTrainedModel class) model from GPTQModel, got model class {type(model).__name__}")
        return model
