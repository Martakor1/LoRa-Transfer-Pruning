"""Profile actual SFT optimizer steps, including Trainer bookkeeping. GPU 2 only."""
import gc
import gzip
import json
import shutil
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import replace
from functools import wraps
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "lora_transfer_pruning").is_dir())))
from experiments.qlora.torchao_exploration.train_qwen_torchao_sft import ExperimentConfig, create_trainer, prepare_packed_datasets, set_seed
from experiments.qlora.torchao_exploration.train_qwen_torchao import load_gptq_model, add_lora
from experiments.qlora.torchao_exploration.replay_qwen_sp_plan import wrap_qwen, apply_saved_plan
from experiments.qlora.torchao_exploration.optimize_qwen_simpruning import skip_redundant_train, cache_flop_count, cache_parameter_iteration
import torch
from transformers import TrainerCallback
from torch.profiler import profile, record_function, ProfilerActivity

OUTPUT = Path(__file__).resolve().parent / 'torchao_trainer_overhead'


class StepPhases(TrainerCallback):
    def __init__(self, trainer, record_cuda, profile_step=4):
        self.trainer = trainer
        self.records = []
        self.active = False
        self.restore = []
        self.prof = None
        self.record_cuda = record_cuda
        self.profile_step = profile_step
        for obj, name, label in [
            (trainer, 'training_step', 'training_step'),
            (trainer, 'compute_loss', 'compute_loss_and_metrics'),
            (trainer.accelerator, 'backward', 'backward'),
            (trainer, '_prepare_inputs', 'prepare_inputs'),
            (trainer, 'floating_point_ops', 'flop_counter'),
            (trainer, '_clip_grad_norm', 'gradient_clipping'),
            (trainer, '_get_grad_norm', 'get_grad_norm'),
            (trainer.model, 'zero_grad', 'zero_grad'),
            (trainer.model, 'train', 'set_train_mode'),
        ]:
            self.wrap(obj, name, label)

    def wrap(self, obj, name, label):
        original = getattr(obj, name)
        @wraps(original)
        def measured(*args, **kwargs):
            if not self.active:
                return original(*args, **kwargs)
            started = time.perf_counter()
            with record_function('phase.' + label):
                value = original(*args, **kwargs)
            self.records.append(dict(step=self.trainer.state.global_step + 1,
                                     phase=label, seconds=time.perf_counter() - started))
            return value
        object.__setattr__(obj, name, measured)
        self.restore.append((obj, name, original))

    def on_step_begin(self, args, state, control, **kwargs):
        self.active = True
        if self.record_cuda and state.global_step == self.profile_step - 1:
            self.prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                                record_shapes=False, with_stack=False, profile_memory=False)
            self.prof.start()

    def on_step_end(self, args, state, control, **kwargs):
        self.active = False
        if self.prof is not None and state.global_step == self.profile_step:
            self.prof.stop()

    def finish(self, directory):
        for obj, name, method in reversed(self.restore):
            object.__setattr__(obj, name, method)
        if self.prof is not None:
            path = directory / f'step{self.profile_step}.chrome.json'
            self.prof.export_chrome_trace(str(path))
            with path.open('rb') as src, gzip.open(str(path)+'.gz', 'wb', compresslevel=1) as dst:
                shutil.copyfileobj(src, dst)
            # No key_averages(): retain raw trace, avoid a second costly event-tree aggregation.
            self.prof = None
        self.restore.clear()
        self.trainer = None


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    set_seed(42)
    model, tokenizer, revision = load_gptq_model()
    model = add_lora(model)
    config = replace(ExperimentConfig(), steps=7, eval_every=100)
    train, evaluation, dataset_info = prepare_packed_datasets(tokenizer, config)
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    initial = [p.detach().cpu().clone() for p in parameters]
    sample = {'input_ids': torch.tensor(train[:8]['input_ids'])}
    results = []
    baseline_weights = None

    def run(name, record_cuda=False, cached_flops=False, compare_weights=False):
        nonlocal baseline_weights
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for p, value in zip(parameters, initial):
                p.copy_(value)
        set_seed(42)
        directory = OUTPUT / name
        directory.mkdir(exist_ok=True)
        trainer, metrics = create_trainer(model, tokenizer, train, evaluation, config, directory)
        # Profile training only; all cases keep the same SFT loss/entropy/accuracy computation.
        trainer.args.eval_strategy = 'no'
        if cached_flops:
            cache_flop_count(trainer, sample)
        phases = StepPhases(trainer, record_cuda)
        trainer.add_callback(phases)
        trainer.train()
        # Never report instrumented/profiler timings as normal throughput.
        keep = {3, 5, 6, 7}
        steps = [r for r in metrics.steps if r['step'] in keep]
        total = sum(r['step_seconds'] for r in steps)
        grouped = defaultdict(list)
        for r in phases.records:
            if r['step'] in keep:
                grouped[r['phase']].append(r['seconds'])
        row = dict(name=name, tokens_per_s=4096 * len(steps) / total,
                   step_s=total/len(steps), peak_GiB=max(r['peak_allocated_GiB'] for r in steps),
                   phases_s={k: statistics.mean(v) for k,v in grouped.items()},
                   steps=metrics.steps, phases=phases.records, history=trainer.state.log_history,
                   total_flos=trainer.state.total_flos,
                   skipped=sum(r['optimizer_step_skipped'] for r in metrics.steps))
        if name == 'tl_sp_cached_train':
            baseline_weights = [p.detach().cpu().clone() for p in parameters]
        elif compare_weights:
            row['adapter_max_delta_vs_sp'] = max((p.detach().cpu()-v).abs().max().item()
                                                  for p,v in zip(parameters, baseline_weights))
        results.append(row)
        (OUTPUT/'results.json').write_text(json.dumps(dict(config=config.__dict__, checkpoint_revision=revision,
                                                         dataset=dataset_info, results=results), indent=2))
        print('RESULT', {k:v for k,v in row.items() if k not in ('steps','phases','history')}, flush=True)
        phases.finish(directory)
        metrics.trainer = None
        del trainer, metrics, phases
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()

    run('hf_sdpa', record_cuda=True)
    model.get_base_model().config._attn_implementation = 'eager'
    run('hf_eager')
    bridge = wrap_qwen(model)
    skip_redundant_train(model)
    run('tl_no_sp_cached_train')
    setup = apply_saved_plan(model, bridge, train)
    (OUTPUT/'pruning_setup.json').write_text(json.dumps(setup, indent=2))
    run('tl_sp_cached_train', record_cuda=True)
    run('tl_sp_cached_flops', cached_flops=True, compare_weights=True)
    original_parameters = cache_parameter_iteration(model)
    run('tl_sp_cached_flops_params', cached_flops=True, compare_weights=True)
    object.__setattr__(model, 'parameters', original_parameters)


if __name__ == '__main__':
    main()
