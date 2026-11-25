from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel


class ModelLoader(Protocol):
    """Load a pretrained model for the preparation pipeline."""

    def load(
        self,
        model_name: str | Path,
        *,
        dtype: torch.dtype | Literal["auto"],
        device_map: str | dict[str, str | int | torch.device] | torch.device | int | None,
        defuse_experts: bool,
        **options: Any,
    ) -> PreTrainedModel:
        """Load a pretrained model from a checkpoint.

        Args:
            model_name: Hugging Face repository ID or path to a local checkpoint.
            dtype: Weight dtype. With "auto", the loader infers it from the
                checkpoint configuration or weights.
            device_map: Model placement during loading. Accepts a placement
                strategy, a single device, or a mapping from module paths to
                devices. None uses the loader's default placement.
            defuse_experts: Whether to split fused mixture-of-experts weights
                into separate expert projections where supported.
            **options: Additional arguments forwarded to the underlying loader,
                such as revision or local_files_only.
        """
        ...
