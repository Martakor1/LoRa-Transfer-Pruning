import torch
from torch import nn
from gptqmodel.nn_modules.qlinear.torch import TorchLinear

class UniversalDequantizer:
    """Read a dense weight in [out_features, in_features] layout."""

    @staticmethod
    def to_dense_weight(layer: nn.Module, device: torch.device | str) -> torch.Tensor:
        """Read a dense Linear weight or dequantize a GPTQ TorchLinear on demand."""
        if isinstance(layer, nn.Linear):
            return layer.weight.to(device)

        if isinstance(layer, TorchLinear):
            scales: torch.Tensor | None = getattr(layer, "scales", None)
            if scales is None:
                raise ValueError("Missing GPTQ scales for dequantization")
            # GPTQ returns [in_features, out_features].
            return layer.dequantize_weight().T.to(
                device, dtype=scales.dtype, memory_format=torch.contiguous_format,
            )

        raise TypeError(f"Cannot read a dense weight from {type(layer).__name__}")
