"""Profile one real training step and sweep batch/context on physical GPU 2.

CUDA_VISIBLE_DEVICES=2 COMPUTE_DTYPE=fp16 .venv/bin/python -u \
    experiments/qlora/slop_experiments/benchmark_qwen_torchao.py

Reuses the TorchAO experiment unchanged. No Trainer/evaluation/data-loading is
timed. Each configuration starts from the same adapters and fresh SGD state.
"""

import argparse
import csv
import gc
import gzip
import json
import math
import os
import shutil
import statistics
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("COMPUTE_DTYPE", "fp16")

import torch
from torch.profiler import ProfilerActivity, profile, record_function

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "lora_transfer_pruning").is_dir())))
from experiments.qlora.torchao_exploration.train_qwen_torchao import (
    COMPUTE_DTYPE, GPTQ_MODEL, HERE, add_lora, load_dataset, load_gptq_model,
)


def make_inputs(tokenizer, max_batch, max_context):
    """Distinct chunks of real text, no padding or random token IDs."""
    cached = sorted(Path(os.environ["HF_HUB_CACHE"]).glob(
        "datasets--Salesforce--wikitext/snapshots/*/wikitext-2-raw-v1/validation-00000-of-00001.parquet"))
    raw = load_dataset("parquet", data_files=str(cached[0]), split="train")
    text = "\n\n".join(text for text in raw["text"] if text.strip())
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    count = max_batch * max_context
    assert len(ids) >= count
    return ids[:count].reshape(max_batch, max_context).cuda()


def training_step(model, batch, optimizer, scaler, annotate=False):
    """Synchronized wall timings, including Python/launch overhead per phase."""
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with record_function("train.forward_and_loss") if annotate else nullcontext():
        with torch.autocast("cuda", dtype=COMPUTE_DTYPE):
            loss = model(**batch, use_cache=False).loss
    torch.cuda.synchronize()
    forward_end = time.perf_counter()
    with record_function("train.backward") if annotate else nullcontext():
        scaler.scale(loss).backward()
    torch.cuda.synchronize()
    backward_end = time.perf_counter()
    with record_function("train.clip_and_optimizer") if annotate else nullcontext():
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), 1.0,
        )
        scaler.step(optimizer)
        previous_scale = scaler.get_scale()
        scaler.update()
    torch.cuda.synchronize()
    end = time.perf_counter()
    result = {
        "forward_s": forward_end - start, "backward_s": backward_end - forward_end,
        "optimizer_s": end - backward_end, "step_s": end - start,
        "loss": loss.item(), "grad_norm": norm.item(),
        "grad_scale": scaler.get_scale(), "update_skipped": scaler.get_scale() < previous_scale,
    }
    return result


def save_profile(model, batch, optimizer, scaler, output):
    """One warmed-up step; shapes/stacks/memory recording disabled to keep it small."""
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False, profile_memory=False, with_stack=False) as prof:
        training_step(model, batch, optimizer, scaler, annotate=True)
    trace = output / "training_b8_s128.chrome.json"
    prof.export_chrome_trace(str(trace))
    with trace.open("rb") as src, gzip.open(str(trace) + ".gz", "wb") as dst:
        shutil.copyfileobj(src, dst)
    rows = []
    for event in prof.key_averages():
        rows.append({"name": event.key, "calls": event.count,
                     "self_cpu_ms": event.self_cpu_time_total / 1000,
                     "cpu_total_ms": event.cpu_time_total / 1000,
                     "self_device_ms": event.self_device_time_total / 1000,
                     "device_total_ms": event.device_time_total / 1000})
    (output / "profile_operators.json").write_text(json.dumps(rows, indent=2))
    print(f"Chrome trace saved: {trace}", flush=True)
    del prof
    gc.collect()
    summarize_profile(output)


def summarize_profile(output):
    """Keep GPU timeline + phase labels; full trace retains the operator tree."""
    trace_path = output / "training_b8_s128.chrome.json"
    trace = json.loads(trace_path.read_text())
    events = trace["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
    groups = defaultdict(list)
    for event in kernels:
        groups[event["name"]].append(event["dur"])
    totals = [{"kernel": name, "calls": len(durations), "total_ms": sum(durations) / 1000,
               "median_us": statistics.median(durations)} for name, durations in groups.items()]
    totals.sort(key=lambda row: row["total_ms"], reverse=True)
    phases = [{"name": e["name"], "duration_s": e["dur"] / 1e6}
              for e in events if e.get("name", "").startswith("train.")
              and e.get("cat") == "user_annotation" and e.get("ph") == "X"]
    summary = {"phases_profiled": phases, "kernel_count": len(kernels),
               "sum_kernel_ms": sum(e["dur"] for e in kernels) / 1000,
               "note": "Profiler slows CPU launches. Kernel sums are not end-to-end wall time; inclusive operator times overlap.",
               "kernels": totals}
    (output / "profile_summary.json").write_text(json.dumps(summary, indent=2))
    compact = []
    for event in events:
        if (event.get("ph") == "M" or event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
                or (event.get("name", "").startswith("train.") and event.get("cat") == "user_annotation")):
            # Long argument strings / redundant launch metadata are in the full trace.
            compact.append(event if event.get("ph") == "M" else
                           {key: value for key, value in event.items() if key != "args"})
        elif (event.get("cat") == "cuda_runtime" and "Synchronize" in event.get("name", "")
              and event.get("dur", 0) >= 100):
            compact.append({key: value for key, value in event.items() if key != "args"})
    trace["traceEvents"] = compact
    compact_path = output / "training_b8_s128.compact.chrome.json"
    compact_path.write_text(json.dumps(trace, separators=(",", ":")))
    with compact_path.open("rb") as src, gzip.open(str(compact_path) + ".gz", "wb") as dst:
        shutil.copyfileobj(src, dst)


def save_results(output, metadata, rows):
    (output / "benchmark.json").write_text(json.dumps({"setup": metadata, "results": rows}, indent=2))
    flat = [{k: v for k, v in row.items() if k not in ("steps", "warmup")} for row in rows]
    if flat:
        keys = list(dict.fromkeys(k for row in flat for k in row))
        with (output / "benchmark.csv").open("w") as f:
            writer = csv.DictWriter(f, keys)
            writer.writeheader()
            writer.writerows(flat)


def plot_results(output, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    batches = sorted({r["batch_size"] for r in rows})
    contexts = sorted({r["context_length"] for r in rows})
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    metrics = [("step_s", "Full step (s)"), ("samples_per_s", "Samples / s"),
               ("input_tokens_per_s", "Input tokens / s"), ("forward_s", "Forward + loss (s)"),
               ("backward_s", "Backward (s)"), ("peak_allocated_GiB", "Peak allocated (GiB)")]
    for ax, (metric, title) in zip(axes.flat, metrics):
        matrix = np.full((len(batches), len(contexts)), np.nan)
        for row in rows:
            if row["status"] == "OK":
                matrix[batches.index(row["batch_size"]), contexts.index(row["context_length"])] = row[metric]
        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        for i in range(len(batches)):
            for j in range(len(contexts)):
                value = matrix[i, j]
                text = f"{value:.2f}" if metric != "input_tokens_per_s" else f"{value:.0f}"
                ax.text(j, i, text if np.isfinite(value) else "OOM/error", ha="center", va="center",
                        color="white", bbox={"facecolor": "black", "alpha": .3, "edgecolor": "none"})
        ax.set(xticks=range(len(contexts)), xticklabels=contexts, yticks=range(len(batches)),
               yticklabels=batches, xlabel="Context length (tokens)", ylabel="Batch size", title=title)
        fig.colorbar(im, ax=ax, shrink=.8)
    fig.suptitle("Qwen3-30B-A3B · TorchAO INT4 + FP16 LoRA · A100 40GB\n"
                 "LoRA layers 45–47; full real-text blocks; warmed synchronized step medians")
    fig.savefig(output / "batch_context_matrix.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--contexts", nargs="+", type=int, default=[64, 128, 256, 512])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "5_speed_vs_batch_size" / "torchao_training_benchmark")
    parser.add_argument("--skip-profile", action="store_true")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model, tokenizer, revision = load_gptq_model()
    model = add_lora(model).train()
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    initial = {name: p.detach().cpu().clone() for name, p in trainable.items()}
    tokens = make_inputs(tokenizer, max(8, *args.batches), max(128, *args.contexts))
    metadata = {
        "checkpoint": GPTQ_MODEL, "revision": revision, "dtype": str(COMPUTE_DTYPE),
        "gpu": torch.cuda.get_device_name(), "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"],
        "torch": torch.__version__, "lora_layers": [45, 46, 47], "lora_rank": 16,
        "trainable_parameters": sum(p.numel() for p in trainable.values()),
        "optimizer": "SGD lr=3e-4 momentum=.9 nesterov, clip=1, AMP GradScaler",
        "gradient_checkpointing": False, "attention": "sdpa", "warmup_steps": args.warmup,
        "measured_steps": args.steps, "cpu_threads": 4,
        "input": "Packed WikiText validation text; exact B*S tokens, no padding. Same prefixes per row.",
        "measurement": "Median of synchronized wall times. Excludes zero_grad, dataloading, reset, profiling. Includes forward/loss, scaled backward, clip/optimizer. No Trainer overhead.",
        "kernel": {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1},
    }

    def reset():
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name, p in trainable.items():
                p.copy_(initial[name])
        torch.manual_seed(0)
        return (torch.optim.SGD(trainable.values(), lr=3e-4, momentum=.9, nesterov=True),
                torch.amp.GradScaler("cuda", enabled=COMPUTE_DTYPE == torch.float16))

    def make_batch(b, s):
        ids = tokens[:b, :s].contiguous()
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}

    if not args.skip_profile:
        optimizer, scaler = reset()
        batch = make_batch(8, 128)
        print("Profile warmup (2 steps)", flush=True)
        for _ in range(2):
            print(training_step(model, batch, optimizer, scaler), flush=True)
        save_profile(model, batch, optimizer, scaler, output)
        model.zero_grad(set_to_none=True)
        del optimizer, scaler, batch
        gc.collect()
        torch.cuda.empty_cache()

    rows = []
    for context in args.contexts:
        for batch_size in args.batches:
            row = {"batch_size": batch_size, "context_length": context, "status": "OK"}
            optimizer, scaler = reset()
            batch = make_batch(batch_size, context)
            print(f"START B={batch_size} S={context}", flush=True)
            try:
                row["warmup"] = [training_step(model, batch, optimizer, scaler) for _ in range(args.warmup)]
                torch.cuda.reset_peak_memory_stats()
                steps = [training_step(model, batch, optimizer, scaler) for _ in range(args.steps)]
                row["steps"] = steps
                for key in ("forward_s", "backward_s", "optimizer_s", "step_s"):
                    row[key] = statistics.median(step[key] for step in steps)
                row["step_min_s"] = min(step["step_s"] for step in steps)
                row["step_max_s"] = max(step["step_s"] for step in steps)
                row["fwd_bwd_s"] = statistics.median(step["forward_s"] + step["backward_s"] for step in steps)
                row["samples_per_s"] = batch_size / row["step_s"]
                row["input_tokens_per_s"] = batch_size * context / row["step_s"]
                row["loss_tokens_per_s"] = batch_size * (context - 1) / row["step_s"]
                row["peak_allocated_GiB"] = torch.cuda.max_memory_allocated() / 2**30
                row["peak_reserved_GiB"] = torch.cuda.max_memory_reserved() / 2**30
                row["skipped_measured_updates"] = sum(step["update_skipped"] for step in steps)
                if row["skipped_measured_updates"] or not all(
                    math.isfinite(step["loss"]) and math.isfinite(step["grad_norm"])
                    for step in steps
                ):
                    row["status"] = "NONFINITE_OR_SKIPPED"
            except torch.cuda.OutOfMemoryError as error:
                row.update(status="OOM", error=str(error))
            rows.append(row)
            save_results(output, metadata, rows)
            print(json.dumps({k: v for k, v in row.items() if k not in ("steps", "warmup")}), flush=True)
            model.zero_grad(set_to_none=True)
            del optimizer, scaler, batch
            gc.collect()
            torch.cuda.empty_cache()
    plot_results(output, rows)
    print(f"DONE: {output}", flush=True)


if __name__ == "__main__":
    main()
