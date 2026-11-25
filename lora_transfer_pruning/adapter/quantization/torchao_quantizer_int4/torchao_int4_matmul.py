import torch
from torch.autograd.function import FunctionCtx


class _Int4MatmulContext(FunctionCtx):
    """Attributes used on the context supplied by autograd."""

    group_size: int
    input_shape: torch.Size
    saved_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class TorchAOInt4Matmul(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: _Int4MatmulContext,
        x: torch.Tensor,
        packed: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        group_size: int,
    ) -> torch.Tensor:
        from torchao.prototype.hqq import triton_mixed_mm

        ctx.save_for_backward(packed, scales, zeros)
        ctx.group_size = group_size
        ctx.input_shape = x.shape
        y = triton_mixed_mm(x.reshape(-1, x.shape[-1]), packed, scales, zeros, group_size)
        return y.reshape(*x.shape[:-1], packed.shape[1])

    @staticmethod
    def backward(
        ctx: _Int4MatmulContext, *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None]:
        from torchao.prototype.hqq import triton_mixed_mm

        (grad_output,) = grad_outputs
        packed, scales, zeros = ctx.saved_tensors
        grad_input = triton_mixed_mm(
            grad_output.reshape(-1, grad_output.shape[-1]).contiguous(),
            packed, scales, zeros, ctx.group_size, transposed=True,
        )
        return grad_input.reshape(ctx.input_shape), None, None, None, None
