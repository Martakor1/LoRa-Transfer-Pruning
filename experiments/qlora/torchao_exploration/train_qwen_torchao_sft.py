"""Actual SFT experiment using the already benchmarked TorchAO INT4 backend.

CUDA_VISIBLE_DEVICES=2 COMPUTE_DTYPE=fp16 .venv/bin/python -u \
    experiments/qlora/torchao_exploration/train_qwen_torchao_sft.py
"""

import json
import math
import os
import time
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("COMPUTE_DTYPE", "fp16")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.qlora.torchao_exploration.train_qwen_torchao import (
    COMPUTE_DTYPE, GPTQ_MODEL, TORCHAO_EXPERIMENT, add_lora, load_gptq_model,
)
import torch
from datasets import Dataset, load_dataset
from transformers import TrainerCallback, default_data_collator, set_seed
from trl import SFTConfig, SFTTrainer

OUTPUT = TORCHAO_EXPERIMENT / "torchao_sft"


@dataclass
class ExperimentConfig:
    steps: int = 100
    batch_size: int = 8
    context_length: int = 512
    train_blocks: int = 1024
    eval_blocks: int = 32
    eval_every: int = 20
    learning_rate: float = 2e-4
    warmup_steps: int = 5
    seed: int = 42


def prepare_packed_datasets(tokenizer, config):
    """Pack official splits independently. Never split after tokenization."""
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.eos_token_id is not None
    snapshot = sorted(Path(os.environ["HF_HUB_CACHE"]).glob(
        "datasets--Salesforce--wikitext/snapshots/*/wikitext-2-raw-v1"))[0]
    result = []
    metadata = {}
    for split, limit in (("train", config.train_blocks), ("validation", config.eval_blocks)):
        raw = load_dataset("parquet", data_files=str(snapshot / f"{split}-00000-of-00001.parquet"), split="train")
        texts = [text for text in raw["text"] if text.strip()]
        tokens = []
        for start in range(0, len(texts), 512):
            encoded = tokenizer(texts[start:start + 512], add_special_tokens=False,
                                truncation=False, padding=False, verbose=False)
            for ids in encoded["input_ids"]:
                tokens.extend(ids)
                tokens.append(tokenizer.eos_token_id)
        length = config.context_length
        usable = len(tokens) // length * length
        blocks = [tokens[i:i + length] for i in range(0, usable, length)]
        assert len(blocks) >= limit, (split, len(blocks), limit)
        data = Dataset.from_dict({"input_ids": blocks})
        data = data.shuffle(seed=config.seed).select(range(limit))
        data = data.map(lambda row: {"labels": row["input_ids"].copy(), "attention_mask": [1] * length})
        result.append(data)
        metadata[split] = {"source": str(snapshot / f"{split}-00000-of-00001.parquet"),
                           "available_blocks": len(blocks), "selected_blocks": limit,
                           "input_tokens": limit * length, "loss_tokens": limit * (length - 1),
                           "discarded_tail_tokens": len(tokens) - usable}
    metadata["packing"] = "EOS after each nonempty text record; concatenate within official split; no padding; retain EOS labels."
    return *result, metadata


class TrainingMetrics(TrainerCallback):
    """Keep standard SFT metrics; add perplexity, step throughput and memory."""
    def __init__(self, config, output):
        self.config = config
        self.output = output
        self.steps = []
        self.trainer = None

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.started = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.started
        self.steps.append({"step": state.global_step, "step_seconds": elapsed,
                           "samples_per_second": self.config.batch_size / elapsed,
                           "tokens_per_second": self.config.batch_size * self.config.context_length / elapsed,
                           "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30,
                           "optimizer_step_skipped": self.trainer.accelerator.optimizer_step_was_skipped})

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "loss" in logs:
            logs["perplexity"] = math.exp(logs["loss"]) if logs["loss"] < 80 else math.inf
            if self.steps:
                logs.update({k: v for k, v in self.steps[-1].items() if k != "step"})
        if "eval_loss" in logs:
            logs["eval_perplexity"] = math.exp(logs["eval_loss"])
        # Trainer appended a copy before invoking callbacks.
        state.log_history[-1].update(logs)
        (self.output / "history.json").write_text(json.dumps(state.log_history, indent=2))


def create_trainer(model, tokenizer, train, evaluation, config, output):
    args = SFTConfig(
        output_dir=str(output / "trainer"), max_steps=config.steps,
        per_device_train_batch_size=config.batch_size, per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=1, gradient_checkpointing=False,
        learning_rate=config.learning_rate, warmup_steps=config.warmup_steps,
        lr_scheduler_type="cosine", max_grad_norm=1.0,
        fp16=True, bf16=False, logging_steps=1, logging_first_step=True,
        eval_strategy="steps", eval_steps=config.eval_every, prediction_loss_only=True,
        save_strategy="no", report_to="none", disable_tqdm=True,
        seed=config.seed, data_seed=config.seed, dataloader_num_workers=0,
        max_length=config.context_length, packing=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        # Keep our attention_mask; default_data_collator does not recreate it.
        remove_unused_columns=False, logging_nan_inf_filter=False,
    )
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=config.learning_rate, weight_decay=0.01)
    metrics = TrainingMetrics(config, output)
    trainer = SFTTrainer(model=model, args=args, train_dataset=train, eval_dataset=evaluation,
                         processing_class=tokenizer, data_collator=default_data_collator,
                         optimizers=(optimizer, None), callbacks=[metrics])
    metrics.trainer = trainer
    return trainer, metrics


def run_experiment(config=None, output=OUTPUT, prepare_model=None):
    config = config or ExperimentConfig()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    assert COMPUTE_DTYPE == torch.float16, "Use COMPUTE_DTYPE=fp16, the benchmarked mode"
    torch.set_num_threads(4)
    set_seed(config.seed)
    model, tokenizer, revision = load_gptq_model()
    model = add_lora(model)
    train, evaluation, dataset_info = prepare_packed_datasets(tokenizer, config)
    setup = {"config": asdict(config), "checkpoint": GPTQ_MODEL, "revision": revision,
             "backend": "GPTQ INT4 repacked for TorchAO fused Triton forward + backward",
             "compute_dtype": str(COMPUTE_DTYPE), "lora_parameter_dtype": "torch.float32",
             "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"], "gpu": torch.cuda.get_device_name(),
             "lora_layers": [45, 46, 47], "lora_rank": 16, "lora_alpha": 32,
             "optimizer": "AdamW, weight_decay=.01, cosine scheduler",
             "pruning": False, "dataset": dataset_info}
    (output / "setup.json").write_text(json.dumps(setup, indent=2))
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    initial = {name: p.detach().cpu().clone() for name, p in trainable.items()}
    assert all(p.dtype == torch.float32 for p in trainable.values())
    trainer, metrics = create_trainer(model, tokenizer, train, evaluation, config, output)
    print("BASELINE validation", flush=True)
    trainer.evaluate()
    before = dict(trainer.state.log_history[-1])
    unpruned_before = None
    if prepare_model is not None:
        unpruned_before = before
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            setup["pruning"] = prepare_model(model, tokenizer, train, trainer, output)
        # Wrapping/instrumenting must not replace the optimizer's parameters.
        assert {id(p) for p in model.parameters() if p.requires_grad} == {id(p) for p in trainable.values()}
        (output / "setup.json").write_text(json.dumps(setup, indent=2))
        print("BASELINE after sim-pruning", flush=True)
        trainer.evaluate()
        before = dict(trainer.state.log_history[-1])
    print("TRAIN", flush=True)
    started = time.perf_counter()
    training = trainer.train()
    elapsed = time.perf_counter() - started
    # Evaluation is normally already performed at the last optimizer step.
    eval_rows = [row for row in trainer.state.log_history if "eval_loss" in row]
    if not eval_rows or eval_rows[-1]["step"] != config.steps:
        trainer.evaluate()
    history = trainer.state.log_history
    # Trainer.train() creates a fresh TrainerState, dropping the step-zero eval.
    if not any(row.get("step") == 0 and "eval_loss" in row for row in history):
        history.insert(0, before)
    after = dict([row for row in history if "eval_loss" in row][-1])
    changed = 0
    max_delta = 0.0
    for name, p in trainable.items():
        delta = (p.detach().cpu() - initial[name]).abs().max().item()
        max_delta = max(max_delta, delta)
        changed += delta > 0
    steady = metrics.steps[2:] or metrics.steps
    total_seconds = sum(row["step_seconds"] for row in steady)
    train_rows = [row for row in history if "loss" in row]
    result = {"setup": setup, "before": before, "after": after, "history": history,
              "train_metrics": training.metrics, "training_wall_seconds_including_eval": elapsed,
              "step_metrics": metrics.steps,
              "steady_samples_per_second": config.batch_size * len(steady) / total_seconds,
              "steady_tokens_per_second": config.batch_size * config.context_length * len(steady) / total_seconds,
              "peak_allocated_GiB": max(row["peak_allocated_GiB"] for row in metrics.steps),
              "adapter_check": {"parameter_tensors": len(trainable), "changed_tensors": changed,
                                "max_abs_update": max_delta},
              "finite_train_metrics": all(math.isfinite(row["loss"]) and math.isfinite(row.get("grad_norm", 0)) for row in train_rows),
              "skipped_updates": sum(row["optimizer_step_skipped"] for row in metrics.steps),
              "eval_loss_improved": after["eval_loss"] < before["eval_loss"]}
    if unpruned_before is not None:
        result["unpruned_before"] = unpruned_before
    (output / "training.json").write_text(json.dumps(result, indent=2))
    (output / "history.json").write_text(json.dumps(history, indent=2))
    print("RESULT", json.dumps({k: v for k, v in result.items() if k not in ("history", "step_metrics", "setup")}, indent=2), flush=True)
    assert changed > 0 and result["finite_train_metrics"], "Training did not produce finite adapter updates"
    return result


def plot_training_history(history):
    """Raw per-step values plus an explicitly labelled train-loss rolling mean."""
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.DataFrame(history)
    metrics = [("loss", "Loss (NLL)"), ("perplexity", "Perplexity"),
               ("mean_token_accuracy", "Next-token accuracy"), ("entropy", "Entropy (nats)"),
               ("grad_norm", "Gradient norm"), ("learning_rate", "Learning rate"),
               ("num_tokens", "Processed input tokens"), ("tokens_per_second", "Training input tokens / s"),
               ("samples_per_second", "Samples / s (train and eval)"), ("step_seconds", "Training step (s)"),
               ("peak_allocated_GiB", "Peak allocated GPU memory (GiB)")]
    fig, axes = plt.subplots(6, 2, figsize=(15, 22), constrained_layout=True)
    for ax, (key, label) in zip(axes.flat, metrics):
        for prefix, title in (("", "Train"), ("eval_", "Validation (fixed subset)")):
            column = prefix + key
            if column not in df:
                continue
            rows = df[["step", column]].dropna()
            if rows.empty:
                continue
            ax.plot(rows.step, rows[column], label=title, marker="o" if prefix else None,
                    alpha=.9 if prefix else .55, linewidth=2 if prefix else 1)
            if key == "loss" and not prefix:
                ax.plot(rows.step, rows[column].rolling(10, min_periods=1).mean(),
                        label="Train rolling mean (10 steps)", linewidth=2)
        ax.set(title=label, xlabel="Optimizer step", ylabel=label)
        ax.grid(alpha=.25)
        ax.legend()
    axes.flat[-1].set_visible(False)
    fig.suptitle("Qwen3-30B-A3B · TorchAO INT4 + LoRA · SFTTrainer", fontsize=16)
    return fig


if __name__ == "__main__":
    result = run_experiment()
    import matplotlib
    matplotlib.use("Agg")
    plot_training_history(result["history"]).savefig(OUTPUT / "training_history.png", dpi=130)
