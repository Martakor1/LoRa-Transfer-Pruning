"""Run on physical GPU 2: CUDA_VISIBLE_DEVICES=2 python .../probe_torchao_int4.py."""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from torchao_int4 import TorchAOInt4Linear, delta, use_fixed_kernel_config


def milliseconds(fn, repeats=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / repeats


if __name__ == "__main__":
    use_fixed_kernel_config()
    torch.manual_seed(0)
    rows = []
    for dtype in (torch.bfloat16, torch.float16):
        for m, k, n in ((64, 2048, 768), (64, 768, 2048), (7, 2048, 768)):
            print("PROBE", dtype, m, k, n, flush=True)
            weight = torch.randn(n, k, device="cuda", dtype=dtype) / k**0.5
            layer = TorchAOInt4Linear(weight)
            ref_weight = layer.reference_weight()
            x = torch.randn(m, k, device="cuda", dtype=dtype, requires_grad=True)
            cotangent = torch.randn(m, n, device="cuda", dtype=dtype)
            y = layer(x)
            dx, = torch.autograd.grad(y, x, cotangent)
            ref_y = F.linear(x, ref_weight)
            ref_dx, = torch.autograd.grad(ref_y, x, cotangent)
            row = {"dtype": str(dtype), "M": m, "K": k, "N": n,
                   "forward": delta(y, ref_y), "backward": delta(dx, ref_dx),
                   "quantization": delta(ref_weight, weight),
                   "fused_forward_ms": milliseconds(lambda: layer(x)),
                   "dequant_forward_ms": milliseconds(lambda: F.linear(x, layer.reference_weight())),
                   "bf16_or_fp16_forward_ms": milliseconds(lambda: F.linear(x, ref_weight))}
            assert torch.isfinite(y).all() and torch.isfinite(dx).all()
            assert row["forward"]["rel_l2"] < 0.01, row
            assert row["backward"]["rel_l2"] < 0.01, row
            rows.append(row)
            print(json.dumps(row), flush=True)
    output = Path(__file__).resolve().parent / "torchao_int4_probe.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(rows, indent=2))
    saved = []
    def record_saved(tensor):
        saved.append({"shape": list(tensor.shape), "dtype": str(tensor.dtype), "bytes": tensor.numel() * tensor.element_size()})
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(record_saved, lambda tensor: tensor):
        y = layer(x)
        torch.autograd.grad(y, x, cotangent)
    assert len(saved) == 3, saved  # packed uint8, scales, zeros -- not dense W or x
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        y = layer(x)
        torch.autograd.grad(y, x, cotangent)
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(output.parent / "torchao_int4_kernel_trace.json"))
    kernels = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    (output.parent / "torchao_int4_autograd.json").write_text(json.dumps({"saved_tensors": saved, "cuda_kernels": kernels}, indent=2))
    print("Saved for backward:", saved, "CUDA kernels:", kernels, flush=True)
