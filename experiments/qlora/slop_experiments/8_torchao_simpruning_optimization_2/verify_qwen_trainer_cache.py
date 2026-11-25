"""Within-one-Trainer ABBA timing; real backward/AdamW, lr=0 for fixed weights."""
import json
import statistics
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "lora_transfer_pruning").is_dir())))
from experiments.qlora.torchao_exploration.train_qwen_torchao_sft import ExperimentConfig, create_trainer, prepare_packed_datasets, set_seed
from experiments.qlora.torchao_exploration.train_qwen_torchao import load_gptq_model, add_lora
from experiments.qlora.torchao_exploration.replay_qwen_sp_plan import wrap_qwen, apply_saved_plan
from experiments.qlora.torchao_exploration.optimize_qwen_simpruning import skip_redundant_train, cache_flop_count, cache_parameter_iteration
from profile_qwen_training_overhead import OUTPUT as PROFILE_OUTPUT, StepPhases
import torch
from transformers import TrainerCallback

OUTPUT = PROFILE_OUTPUT / 'paired_cache'


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    set_seed(42)
    model, tokenizer, _ = load_gptq_model()
    model = add_lora(model)
    config = replace(ExperimentConfig(), steps=15, learning_rate=0., warmup_steps=0, eval_every=100)
    train, evaluation, _ = prepare_packed_datasets(tokenizer, config)
    # One batch per epoch: same eight text blocks, shuffled by the Trainer.
    train = train.select(range(config.batch_size))
    bridge = wrap_qwen(model)
    apply_saved_plan(model, bridge, train)
    skip_redundant_train(model)
    trainer, metrics = create_trainer(model, tokenizer, train, evaluation, config, OUTPUT)
    trainer.args.eval_strategy = 'no'
    original_parameters = cache_parameter_iteration(model)
    cached_parameters = model.parameters
    original_flops = cache_flop_count(trainer, {'input_ids': torch.tensor(train[:]['input_ids'])})
    cached_flops = trainer.floating_point_ops
    schedule = ['warmup', 'warmup'] + ['baseline', 'cached', 'cached', 'baseline'] * 3 + ['profile_cached']
    mode = 'warmup'

    def flops(inputs):
        return (original_flops if mode in ('warmup', 'baseline') else cached_flops)(inputs)
    trainer.floating_point_ops = flops

    class Switch(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            nonlocal mode
            mode = schedule[state.global_step]
            object.__setattr__(model, 'parameters',
                               original_parameters if mode in ('warmup', 'baseline') else cached_parameters)

    trainer.add_callback(Switch())
    phases = StepPhases(trainer, record_cuda=True, profile_step=15)
    trainer.add_callback(phases)
    trainer.train()
    summaries = {}
    for name in ('baseline', 'cached'):
        steps = [s for s in metrics.steps if schedule[s['step']-1] == name]
        records = [r for r in phases.records if schedule[r['step']-1] == name]
        elapsed = sum(s['step_seconds'] for s in steps)
        summaries[name] = dict(step_s=elapsed/len(steps), tokens_per_s=4096*len(steps)/elapsed,
                               samples_per_s=8*len(steps)/elapsed,
                               peak_GiB=max(s['peak_allocated_GiB'] for s in steps),
                               phases_s={key: statistics.mean(r['seconds'] for r in records if r['phase'] == key)
                                         for key in {r['phase'] for r in records}})
    result = dict(config=config.__dict__, schedule=schedule, summaries=summaries,
                  steps=metrics.steps, phases=phases.records, history=trainer.state.log_history)
    (OUTPUT/'results.json').write_text(json.dumps(result, indent=2))
    print('PAIRED RESULT', summaries, flush=True)
    phases.finish(OUTPUT)
    object.__setattr__(model, 'parameters', original_parameters)


if __name__ == '__main__':
    main()
