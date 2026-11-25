from typing import cast

import torch
from torch import nn

from .torchao_int4_matmul import TorchAOInt4Matmul


class TorchAOInt4Linear(nn.Module):
    """Frozen packed linear module with weight with optional unquantized bias.
    It is not nn.Linear and not TorchLinear from GPTQ, and can't be inherited from them!
    """

    qweight: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    bias: nn.Parameter | None
    compute_dtype: torch.dtype

    def __init__(
        self,
        weight: torch.Tensor,
        group_size: int = 128,
        optimize: bool = True,
        *,
        bias: nn.Parameter | None = None,
    ) -> None:
        from torchao.prototype.hqq import pack_2xint4
        from torchao.quantization.quant_primitives import _choose_qparams_and_quantize_affine_hqq

        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.group_size = group_size
        self.compute_dtype = weight.dtype
        q, scales, zeros, _ = _choose_qparams_and_quantize_affine_hqq(
            weight, nbits=4, group_size=group_size, axis=1,
            compute_dtype=self.compute_dtype, device=str(weight.device),
            optimize=optimize, raw_output=True,
        )
        # Keep the column-major layout required by the TorchAO 0.17 prototype.
        self.register_buffer("qweight", pack_2xint4(q.T))
        # Clone HQQ metadata outside inference mode so backward can save it.
        self.register_buffer("scales", scales.reshape(self.out_features, -1).T.clone())
        self.register_buffer("zeros", zeros.reshape(self.out_features, -1).T.clone())
        self.register_parameter("bias", bias)

    @classmethod
    def from_packed(
        cls,
        tensors: dict[str, torch.Tensor],
        device: torch.device | str = "cuda",
        *,
        bias: nn.Parameter | None = None,
    ) -> "TorchAOInt4Linear":
        module = cls.__new__(cls)
        nn.Module.__init__(module)
        for name, tensor in tensors.items():
            module.register_buffer(name, tensor.to(device))
        module.in_features = module.qweight.shape[0] * 2
        module.out_features = module.qweight.shape[1]
        module.group_size = module.in_features // module.scales.shape[0]
        module.compute_dtype = module.scales.dtype
        module.register_parameter("bias", bias)
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = cast(
            torch.Tensor,
            TorchAOInt4Matmul.apply(
                x.to(self.compute_dtype), self.qweight, self.scales, self.zeros, self.group_size,
            ),
        )
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output.to(x.dtype)

    def reference_weight(self) -> torch.Tensor:
        """Materialize the weight for numerical checks."""
        q = torch.stack((self.qweight & 15, self.qweight >> 4), dim=1).flatten(0, 1)
        scales = self.scales.repeat_interleave(self.group_size, dim=0)
        zeros = self.zeros.repeat_interleave(self.group_size, dim=0)
        return ((q.float() - zeros.float()) * scales.float()).T.to(self.scales.dtype)
