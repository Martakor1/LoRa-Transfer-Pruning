from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear


class TorchAORepackerInt4:
    """Translate compatible GPTQ INT4 buffers into the our custom TorchAO layout."""

    def __init__(self, group_size: int) -> None:
        self._group_size = group_size

    def try_repack(self, layer: TorchLinear, device: torch.device) -> dict[str, torch.Tensor] | None:
        """Return packed TorchAO buffers, or None when the weights need requantization."""
        qweight: torch.Tensor | None = getattr(layer, "qweight", None)
        qzeros: torch.Tensor | None = getattr(layer, "qzeros", None)
        scales: torch.Tensor | None = getattr(layer, "scales", None)
        g_idx: torch.Tensor | None = getattr(layer, "g_idx", None)
        if qweight is None or qzeros is None or scales is None or g_idx is None:
            raise ValueError("Missing packed GPTQ weights")

        groups = layer.in_features // self._group_size
        if (
            layer.bits != 4
            or layer.group_size != self._group_size
            or layer.qzero_format() != 2
            or layer.out_features % 8
            or qweight.dtype != torch.int32
            or qzeros.dtype != torch.int32
            or qweight.shape != (layer.in_features // 8, layer.out_features)
            or qzeros.shape != (groups, layer.out_features // 8)
            or scales.shape != (groups, layer.out_features)
        ):
            return None
        ordered_groups = torch.arange(layer.in_features, device=g_idx.device) // self._group_size
        if not torch.equal(g_idx, ordered_groups):
            return None

        from torchao.prototype.hqq import pack_2xint4

        shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=device)
        codes = ((qweight.to(device)[:, None, :] >> shifts[None, :, None]) & 15).flatten(0, 1)
        # The Torch backend has already converted GPTQ v1 zero points to v2.
        zeros = ((qzeros.to(device)[:, :, None] >> shifts) & 15).flatten(1)
        return {
            "qweight": pack_2xint4(codes.to(torch.uint8).T.contiguous().T),
            "scales": scales.to(device).T.contiguous().T,
            "zeros": zeros.to(scales.dtype).T.contiguous().T,
        }
