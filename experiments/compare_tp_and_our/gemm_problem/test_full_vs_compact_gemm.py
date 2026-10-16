import argparse

import torch
import torch.nn.functional as F


DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def statistics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | int | bool]:
    delta = candidate - reference
    return {
        "equal": torch.equal(reference, candidate),
        "different": torch.count_nonzero(reference != candidate).item(),
        "max": delta.abs().max().item(),
        "mean": delta.abs().mean().item(),
        "rmse": delta.square().mean().sqrt().item(),
    }


def print_statistics(name: str, values: dict[str, float | int | bool]) -> None:
    print(name)
    print(f"  equal:     {values['equal']}")
    print(f"  different: {values['different']}")
    print(f"  max:       {values['max']:.8e}")
    print(f"  mean:      {values['mean']:.8e}")
    print(f"  rmse:      {values['rmse']:.8e}")


def run_experiment(
    dtype: torch.dtype,
    device: torch.device,
    batch_size: int,
    out_features: int,
    compact_k: int,
    full_k: int,
    gap_start: int,
    repeats: int,
    seed: int,
) -> None:
    generator = torch.Generator(device=device).manual_seed(seed)
    x_compact = torch.randn(
        batch_size, compact_k, device=device, dtype=dtype, generator=generator
    )
    w_compact = torch.randn(
        out_features, compact_k, device=device, dtype=dtype, generator=generator
    )

    gap_size = full_k - compact_k
    keep = torch.cat(
        (
            torch.arange(gap_start, device=device),
            torch.arange(gap_start + gap_size, full_k, device=device),
        )
    )

    x_interspersed = torch.zeros(batch_size, full_k, device=device, dtype=dtype)
    w_interspersed = torch.zeros(out_features, full_k, device=device, dtype=dtype)
    x_interspersed.index_copy_(-1, keep, x_compact)
    w_interspersed.index_copy_(1, keep, w_compact)

    x_trailing = F.pad(x_compact, (0, gap_size)).contiguous()
    w_trailing = F.pad(w_compact, (0, gap_size)).contiguous()

    # Warm up cuBLAS and the selected kernels for every shape.
    for _ in range(3):
        F.linear(x_compact, w_compact)
        F.linear(x_interspersed, w_interspersed)
        F.linear(x_trailing, w_trailing)
    torch.cuda.synchronize(device)

    worst_interspersed = None
    worst_trailing = None
    with torch.inference_mode():
        for _ in range(repeats):
            compact = F.linear(x_compact, w_compact)
            interspersed = F.linear(x_interspersed, w_interspersed)
            trailing = F.linear(x_trailing, w_trailing)

            current_interspersed = statistics(compact, interspersed)
            current_trailing = statistics(compact, trailing)
            if worst_interspersed is None or current_interspersed["max"] > worst_interspersed["max"]:
                worst_interspersed = current_interspersed
            if worst_trailing is None or current_trailing["max"] > worst_trailing["max"]:
                worst_trailing = current_trailing

    print(f"\n{dtype} (compact K={compact_k}, full K={full_k}, internal gap at {gap_start})")
    print_statistics("compact vs full with internal zeros", worst_interspersed)
    print_statistics("compact vs full with trailing zeros", worst_trailing)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare compact GEMM with equivalent zero-padded GEMM.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out-features", type=int, default=4096)
    parser.add_argument("--compact-k", type=int, default=3968)
    parser.add_argument("--full-k", type=int, default=4096)
    parser.add_argument("--gap-start", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this GEMM experiment.")
    if not 0 <= args.gap_start <= args.compact_k:
        raise ValueError("gap-start must be between 0 and compact-k.")
    if args.full_k <= args.compact_k:
        raise ValueError("full-k must be greater than compact-k.")

    device = torch.device(args.device)
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"Float32 matmul precision: {torch.get_float32_matmul_precision()}")
        for dtype in DTYPES:
            run_experiment(
                dtype=dtype,
                device=device,
                batch_size=args.batch_size,
                out_features=args.out_features,
                compact_k=args.compact_k,
                full_k=args.full_k,
                gap_start=args.gap_start,
                repeats=args.repeats,
                seed=args.seed,
            )
    finally:
        torch.set_float32_matmul_precision(previous_precision)


if __name__ == "__main__":
    main()
