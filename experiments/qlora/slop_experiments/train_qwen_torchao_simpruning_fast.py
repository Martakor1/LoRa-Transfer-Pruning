"""Keep TL eager + original SP math; skip redundant tree traversals.

Default is the same 100-step run. --steps 6 is a short actual-training check.
--cache-bookkeeping also caches Trainer's parameter traversal/FLOP estimate.
The saved mask plan comes from LocalPruning; rebuild it with the original
simpruning script whenever the model or pruning task changes.
"""
import argparse
import sys
from pathlib import Path
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_qwen_torchao_sft import ExperimentConfig, run_experiment, plot_training_history
from train_qwen_torchao import HERE
from replay_qwen_sp_plan import prepare_saved_plan
from optimize_qwen_simpruning import skip_redundant_train, cache_flop_count, cache_parameter_iteration
import torch


def prepare_fast(model, tokenizer, train, trainer, output, profile_python=False, cache_bookkeeping=False):
    setup = prepare_saved_plan(model, tokenizer, train, trainer, output)
    params = [p for p in model.parameters() if p.requires_grad]
    tokens = torch.tensor(train[0]['input_ids'][:128], device='cuda')[None]

    def probe():
        model.eval()
        with torch.autocast('cuda', dtype=torch.float16):
            result = model(input_ids=tokens, labels=tokens, use_cache=False)
        gradients = torch.autograd.grad(result.loss * 1024, params, allow_unused=True)
        gradient = torch.cat([(g.detach().float().cpu() if g is not None
                               else torch.zeros_like(p, device='cpu')).flatten()
                              for p, g in zip(params, gradients)])
        return result.logits.detach().float().cpu(), gradient

    reference, reference_grad = probe()
    # The original fast-path change is this call. It does not change
    # attention, RMSNorm, masks, weights, precision or quantized kernels.
    original_train = skip_redundant_train(model)
    candidate, candidate_grad = probe()
    setup['optimization'] = dict(
        skip_redundant_train=True,
        attention='unchanged TL eager', rmsnorm='unchanged dynamic correction',
        logits_max_abs=(candidate-reference).abs().max().item(),
        gradient_max_abs=(candidate_grad-reference_grad).abs().max().item(),
    )
    print('FAST PATH CHECK', setup['optimization'], flush=True)
    assert torch.equal(reference, candidate) and torch.equal(reference_grad, candidate_grad)
    # Restore normal recursion before independently changing a child's mode.
    object.__setattr__(model, '_sp_original_train', original_train)
    if cache_bookkeeping:
        object.__setattr__(model, '_sp_original_parameters', cache_parameter_iteration(model))
        trainer._sp_original_floating_point_ops = cache_flop_count(trainer, {'input_ids': tokens})
        setup['optimization']['cached_parameter_iteration'] = True
        setup['optimization']['cached_flop_parameter_count'] = True
    if not profile_python:
        return setup
    # Profile one real Trainer.training_step (Python overhead, not CUDA kernels).
    # It is excluded from steady timings, which discard the first two steps.
    import cProfile
    import pstats
    original_step = trainer.training_step
    profiled = False

    def step(*args, **kwargs):
        nonlocal profiled
        if profiled:
            return original_step(*args, **kwargs)
        profiled = True
        profiler = cProfile.Profile()
        result = profiler.runcall(original_step, *args, **kwargs)
        profiler.dump_stats(str(output / 'training_step.pstats'))
        with (output / 'python_profile.txt').open('w') as stream:
            pstats.Stats(profiler, stream=stream).sort_stats('tottime').print_stats(60)
        return result

    trainer.training_step = step
    return setup


def restore_training_mode_method(model):
    object.__setattr__(model, 'train', model._sp_original_train)
    model.train(model.training)


def restore_training_shortcuts(model, trainer):
    """Call before changing architecture or controlling individual child modes."""
    restore_training_mode_method(model)
    if hasattr(model, '_sp_original_parameters'):
        object.__setattr__(model, 'parameters', model._sp_original_parameters)
    if hasattr(trainer, '_sp_original_floating_point_ops'):
        trainer.floating_point_ops = trainer._sp_original_floating_point_ops


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--profile-python', action='store_true')
    parser.add_argument('--cache-bookkeeping', action='store_true')
    args = parser.parse_args()
    output = Path(__file__).resolve().parent / '7_torchao_simpruing_optimization' / ('torchao_sft_simpruning_fast' if args.steps == 100 else 'torchao_sp_fast_smoke')
    if args.cache_bookkeeping:
        output = Path(__file__).resolve().parent / '8_torchao_simpruning_optimization_2' / ('torchao_sft_simpruning_cached' if args.steps == 100 else 'torchao_sp_cached_smoke')
    config = replace(ExperimentConfig(), steps=args.steps)
    from functools import partial
    result = run_experiment(config, output=output,
                            prepare_model=partial(prepare_fast, profile_python=args.profile_python,
                                                  cache_bookkeeping=args.cache_bookkeeping))
    import matplotlib
    matplotlib.use('Agg')
    plot_training_history(result['history']).savefig(output / 'training_history.png', dpi=130)
