from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from torch import nn

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel


LayerFilter = Callable[[nn.Module, str], bool]
'''Receives a dense or packed linear layer and its qualified path in the model.
Return True to include the layer in quantization.'''


class ModelQuantizer(Protocol):
    """Quantize a pretrained model before injecting LoRA."""

    def quantize(
        self,
        model: PreTrainedModel,
        *,
        device: torch.device | str | None = None,
        filter_fn: LayerFilter | None = None,
    ) -> PreTrainedModel:
        """Quantize eligible linear layers and return the prepared model.

        Args:
            model: Pretrained base model before LoRA injection.
            device: Device on which to perform quantization. None uses the
                current device of each layer's weights.
            filter_fn: Predicate called with each eligible linear layer and its
                qualified module path. True selects the layer for quantization.
                None selects all eligible layers.
        """
        ...
