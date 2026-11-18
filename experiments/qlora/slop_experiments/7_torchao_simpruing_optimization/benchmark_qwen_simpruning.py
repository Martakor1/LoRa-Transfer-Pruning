"""Paired six-step SFT timings, short gradient checks and Chrome profiles on GPU 2."""
import gc
import gzip
import json
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train_qwen_torchao_sft import (
    ExperimentConfig, prepare_packed_datasets, create_trainer, set_seed,
)
from train_qwen_torchao import load_gptq_model, add_lora, HERE
from train_qwen_torchao_simpruning import prepare_simpruning
from optimize_qwen_simpruning import replace_rms_hooks, prepare_async_rmsnorm, use_native_sdpa
from lora_transfer_pruning.core.pruning_instrumentor import PruningInstrumentor
from benchmark_qwen_torchao import training_step
import torch
from torch.profiler import profile, ProfilerActivity

OUTPUT = Path(__file__).resolve().parent / 'torchao_sp_optimization'


def save_profile(model, batch, path):
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4)
    scaler = torch.amp.GradScaler('cuda')
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False, with_stack=False) as prof:
        training_step(model, batch, optimizer, scaler, annotate=True)
    prof.export_chrome_trace(str(path))
    with path.open('rb') as src, gzip.open(str(path) + '.gz', 'wb') as dst:
        shutil.copyfileobj(src, dst)
    rows = [dict(name=e.key, calls=e.count, self_cpu_ms=e.self_cpu_time_total / 1000,
                 cpu_ms=e.cpu_time_total / 1000, self_device_ms=e.self_device_time_total / 1000)
            for e in prof.key_averages()]
    path.with_suffix('.operators.json').write_text(json.dumps(rows, indent=2))
    del prof, optimizer, scaler
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()


def main(extra=False):
    global OUTPUT
    if extra:
        OUTPUT = Path(__file__).resolve().parent / 'torchao_sp_optimization_extra'
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    set_seed(42)
    model, tokenizer, _ = load_gptq_model()
    model = add_lora(model)
    config = replace(ExperimentConfig(), steps=6, eval_every=100)
    train, evaluation, _ = prepare_packed_datasets(tokenizer, config)
    params = [p for p in model.parameters() if p.requires_grad]
    initial = [p.detach().cpu().clone() for p in params]
    batch = {k: torch.tensor(train[:8][k], device='cuda') for k in ('input_ids', 'labels', 'attention_mask')}
    check_batch = {k: v[:1, :128] for k, v in batch.items()}
    rows = []
    checks = []

    def reset():
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for p, value in zip(params, initial):
                p.copy_(value)
        set_seed(42)

    def check():
        reset()
        model.eval()
        with torch.autocast('cuda', dtype=torch.float16):
            output = model(**check_batch, use_cache=False)
        grads = torch.autograd.grad(output.loss * 1024, params, allow_unused=True)
        gradient = torch.cat([(g.detach().float().cpu() if g is not None else torch.zeros_like(p, device='cpu'))
                              .flatten() for p, g in zip(params, grads)])
        return output.logits.detach().float().cpu(), gradient, output.loss.item()

    def run_case(name):
        reset()
        model.train()
        start = time.perf_counter()
        model.train()
        train_mode_seconds = time.perf_counter() - start
        print('model.train() seconds:', train_mode_seconds, flush=True)
        trainer, metrics = create_trainer(model, tokenizer, train, evaluation, config, OUTPUT / name)
        (OUTPUT / name).mkdir(parents=True, exist_ok=True)
        trainer.train()
        seconds = sum(r['step_seconds'] for r in metrics.steps[2:])
        row = dict(name=name, tokens_per_second=4096 * 4 / seconds,
                   train_mode_seconds=train_mode_seconds,
                   samples_per_second=8 * 4 / seconds, step_seconds=seconds / 4,
                   peak_GiB=max(r['peak_allocated_GiB'] for r in metrics.steps),
                   skipped=sum(r['optimizer_step_skipped'] for r in metrics.steps),
                   steps=metrics.steps, history=trainer.state.log_history)
        rows.append(row)
        (OUTPUT / 'benchmark.json').write_text(json.dumps(dict(results=rows, checks=checks), indent=2))
        print('CASE', {k: v for k, v in row.items() if k not in ('steps', 'history')}, flush=True)
        metrics.trainer = None
        del trainer, metrics
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()

    run_case('hf_no_sp')
    reset()
    trainer, metrics = create_trainer(model, tokenizer, train, evaluation, config, OUTPUT / 'preparation')
    (OUTPUT / 'preparation').mkdir(parents=True, exist_ok=True)
    with torch.random.fork_rng(devices=[0]):
        if extra:
            from replay_qwen_sp_plan import prepare_saved_plan
            setup = prepare_saved_plan(model, tokenizer, train, trainer, OUTPUT / 'preparation')
        else:
            setup = prepare_simpruning(model, tokenizer, train, trainer, OUTPUT / 'preparation')
    (OUTPUT / 'setup.json').write_text(json.dumps(setup, indent=2))
    metrics.trainer = None
    del trainer, metrics
    ref_logits, ref_grad, ref_loss = check()
    reference_name = 'tl_sp_original'
    run_case('tl_sp_original')
    if not extra:
        save_profile(model, batch, OUTPUT / 'original_b8_s512.chrome.json')

    def compare(name):
        logits, gradient, loss = check()
        gd = gradient - ref_grad
        checks.append(dict(name=name, reference=reference_name, ref_loss=ref_loss, loss=loss,
                           logits_max_abs=(logits-ref_logits).abs().max().item(),
                           logits_rel_l2=((logits-ref_logits).norm()/ref_logits.norm()).item(),
                           gradient_max_abs=gd.abs().max().item(),
                           gradient_rel_l2=(gd.norm()/ref_grad.norm()).item(),
                           finite=bool(torch.isfinite(gradient).all())))
        print('CHECK', checks[-1], flush=True)

    if extra:
        from optimize_qwen_simpruning import skip_redundant_train
        skip_redundant_train(model)
        compare('cached_train')
        run_case('cached_train')
        use_native_sdpa(model)
        model.get_base_model().config._attn_implementation = 'eager'
        compare('cached_train_native_eager')
        run_case('cached_train_native_eager')
        model.get_base_model().config._attn_implementation = 'sdpa'
        compare('cached_train_native_sdpa')
        run_case('cached_train_native_sdpa')
        ref_logits, ref_grad, ref_loss = check()
        reference_name = 'cached_train_native_sdpa'
        from optimize_qwen_simpruning import bypass_inactive_bridges
        restored = bypass_inactive_bridges(model)
        print('Bypassed empty bridges:', len(restored), flush=True)
        compare('cached_train_native_sdpa_skip_empty')
        run_case('cached_train_native_sdpa_skip_empty')
        save_profile(model, batch, OUTPUT / 'optimized_b8_s512.chrome.json')
        return

    print('RMS modules:', replace_rms_hooks(model, prepare_async_rmsnorm), flush=True)
    compare('async_rms')
    run_case('async_rms')
    replace_rms_hooks(model, PruningInstrumentor.prepare_rms_norm_for_pruning_dynamically)
    print('Native attention modules:', use_native_sdpa(model), flush=True)
    compare('native_sdpa')
    run_case('native_sdpa')
    replace_rms_hooks(model, prepare_async_rmsnorm)
    compare('native_sdpa_async_rms')
    run_case('native_sdpa_async_rms')
    save_profile(model, batch, OUTPUT / 'optimized_b8_s512.chrome.json')


if __name__ == '__main__':
    main(extra='--extra' in sys.argv)
