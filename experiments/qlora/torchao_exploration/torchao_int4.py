"""Frozen HQQ INT4 Linear: TorchAO's fused Triton GEMM in both directions.

Experimental, tied to torchao 0.17's prototype API. No full weight is unpacked
during training. LoRA parameters are handled by PEFT, not by this module.
"""

from lora_transfer_pruning.adapter.quantization.torchao_quantizer_int4 import TorchAOInt4Linear


def delta(actual, expected):
    diff = actual.float() - expected.float()
    return {"max_abs": diff.abs().max().item(), "rmse": diff.square().mean().sqrt().item(),
            "rel_l2": (diff.norm() / expected.float().norm().clamp_min(1e-12)).item()}


def use_fixed_kernel_config():
    """Select one stock configuration: avoid retuning every MoE token count.

    This changes launch tuning, not the kernel or model. All Qwen input/output
    widths are multiples of 128; arbitrary token counts remain masked by kernel.
    """
    from torchao.prototype.hqq.kernels import mixed_mm_kernel_compute_bound
    mixed_mm_kernel_compute_bound.configs = [config for config in mixed_mm_kernel_compute_bound.configs
        if config.kwargs == {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1}]
    assert len(mixed_mm_kernel_compute_bound.configs) == 1
