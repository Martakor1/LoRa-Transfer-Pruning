"""Qwen INT4: two SP adapters jointly versus independent fresh-model runs.

Uses the existing TorchAO loader, LocalPruning, PruningInstrumentor,
MultiAdapterPruningTrainer and PruningLens. No installed-package patches.
"""
import argparse
import copy
import gc
import hashlib
import json
import math
import time
import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

# Establish the existing experiment's environment before importing HF libraries.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.qlora.torchao_exploration.train_qwen_torchao_sft import ExperimentConfig, prepare_packed_datasets, TrainingMetrics
from experiments.qlora.torchao_exploration.train_qwen_torchao import load_gptq_model, add_lora
from experiments.qlora.torchao_exploration.train_qwen_torchao_simpruning import (
    TorchAOGraphOnlyPruner, TorchAOInt4Linear, LocalPruning, ModelPruneTask,
    GroupPruneTask, iter_pruned_linear_hooks, audit_masks, LAYERS, LinearBridge,
)
from experiments.qlora.torchao_exploration.replay_qwen_sp_plan import wrap_qwen
from experiments.qlora.torchao_exploration.optimize_qwen_simpruning import skip_redundant_train, cache_parameter_iteration, cache_flop_count
from lora_transfer_pruning.core.multi_adapter_pruning_trainer import MultiAdapterPruningTrainer
from lora_transfer_pruning.core.pruning_instrumentor import PruningInstrumentor
from lora_transfer_pruning.core.torch_pruning_group_mapper import TorchPruningGroupMapper
from pruning_lens.adapter.transfer_pruning.transfer_pruning_tracer import TransferPruningTracer
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import default_data_collator, set_seed
from trl import SFTConfig

RATIOS = {'prune20': .2, 'prune80': .8}


def adapter_parameters(model, adapter):
    return dict(sorted((name.replace(f'.{adapter}.', '.ADAPTER.'), p)
                       for name, p in model.named_parameters()
                       if f'.{adapter}.' in name and ('.lora_A.' in name or '.lora_B.' in name)))


def make_tasks(bridge):
    """Nested, seed-fixed masks; head coordinates closed under rotate_half."""
    config = bridge.blocks[45].attn._original_component.config
    head_dim = bridge.blocks[45].attn._original_component.head_dim
    width = bridge.blocks[45].mlp.experts[0].up_proj.out_features
    generator = torch.Generator().manual_seed(142)
    head_order = torch.randperm(head_dim // 2, generator=generator)
    mlp_order = torch.randperm(width, generator=generator)
    tasks, description = {}, {}
    for name, ratio in RATIOS.items():
        half = head_order[:round(head_dim / 2 * ratio)]
        heads = torch.cat((half, half + head_dim // 2)).sort().values.tolist()
        mlp = mlp_order[:round(width * ratio)].sort().values.tolist()
        tasks[name] = ModelPruneTask({
            **{f'blocks.{layer}.attn.q': GroupPruneTask(None, heads) for layer in LAYERS},
            **{f'blocks.{layer}.mlp.experts.{i}.up_proj': GroupPruneTask(None, mlp)
               for layer in LAYERS for i in range(len(bridge.blocks[layer].mlp.experts))},
        }, name=name)
        description[name] = dict(requested_ratio=ratio, head_dim=head_dim,
                                 head_indices=heads, mlp_indices=mlp,
                                 head_fraction=len(heads)/head_dim, mlp_fraction=len(mlp)/width,
                                 query_heads=config.num_attention_heads, layers=LAYERS)
    return tasks, description


def prepare(root, cfg):
    """One common mask/initialization stage, excluded from all training arms."""
    started = time.perf_counter()
    set_seed(cfg.seed)
    model, tokenizer, revision = load_gptq_model()
    model = add_lora(model, 'prune20')
    train, evaluation, dataset_info = prepare_packed_datasets(tokenizer, cfg)
    torch.save({'train': train.to_dict(), 'eval': evaluation.to_dict()}, root/'data.pt')
    bridge = wrap_qwen(model)
    initial = {k:p.detach().cpu().clone() for k,p in adapter_parameters(model, 'prune20').items()}
    torch.save(initial, root/'initial_adapter.pt')
    tasks, description = make_tasks(bridge)
    tokens = torch.tensor(train[0]['input_ids'][:16], device='cuda')[None]
    local = LocalPruning(bridge, tokens, peft_model=model,
                         customized_pruners={TorchAOInt4Linear: TorchAOGraphOnlyPruner()})
    # Architecture is now fixed; avoid repeating alias-tree traversal for every group.
    saved = [(m, cache_parameter_iteration(m)) for m in (bridge, bridge.original_model)]
    manifests = {}
    for name, task in tasks.items():
        print('BUILD PLAN', name, flush=True)
        plan = local.get_torch_pruning_groups_and_structural_setups(task)
        rows = [{'hook': point.name, 'removed_indices': list(map(int, ids))}
                for point, ids in iter_pruned_linear_hooks(bridge, plan.groups)]
        manifests[name] = rows
        assert len(plan.groups) == 387
        print('PLAN READY', name, len(plan.groups), len(rows), flush=True)
        del plan
    for module, method in saved:
        object.__setattr__(module, 'parameters', method)
    (root/'plans.json').write_text(json.dumps(manifests, indent=2))
    (root/'preparation.json').write_text(json.dumps(dict(
        seconds=time.perf_counter()-started, config=asdict(cfg), revision=revision,
        masks=description, dataset=dataset_info,
        data_sha256=hashlib.sha256((root/'data.pt').read_bytes()).hexdigest(),
    ), indent=2))


def apply_plan(model, bridge, rows, adapter):
    """Same outer-boundary replay as the optimized single-adapter experiment."""
    boundaries = []
    for row in rows:
        name, direction = row['hook'].rsplit('.', 1)
        module = bridge.get_submodule(name)
        boundaries.append((getattr(module, direction), row['removed_indices']))
        if direction == 'hook_in':
            assert name.endswith(('.attn.o', '.down_proj'))
            continue  # Mapper's rule: upstream outputs already zero these inputs.
        assert direction == 'hook_out'
        PruningInstrumentor.prepare_linear_for_pruning(
            module, None, torch.tensor(row['removed_indices'], device='cuda'), rescale=False,
            active_adapters_source=lambda: model.active_adapters, adapter_name=adapter)
    return boundaries


class ExperimentTrainer(MultiAdapterPruningTrainer):
    def __init__(self, *args, adapter_groups, **kwargs):
        self.adapter_groups = adapter_groups
        self.shared_prefix = None
        self.eval_wall_seconds = 0.
        super().__init__(*args, **kwargs)

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.shared_prefix is None:
            return super().training_step(model, inputs, num_items_in_batch)
        # The parent runs both adapters on these SAME inputs, then backprops each.
        self.shared_prefix.enabled = True
        self.shared_prefix.reused_calls = 0
        try:
            result = super().training_step(model, inputs, num_items_in_batch)
            assert self.shared_prefix.reused_calls == min(LAYERS)
            return result
        finally:
            self.shared_prefix.enabled = False
            self.shared_prefix.cached = None

    def get_train_dataloader(self):
        # Dataset has already been shuffled once and saved; same order in every arm.
        return self.accelerator.prepare(DataLoader(
            self.train_dataset, batch_size=self.args.per_device_train_batch_size,
            shuffle=False, collate_fn=default_data_collator, drop_last=True))

    def _clip_grad_norm(self, model):
        # One unscale, then independent clipping: no coupling of adapter gradients.
        self.accelerator.unscale_gradients(self.optimizer)
        norms = []
        for name, parameters in self.adapter_groups.items():
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            assert torch.isfinite(norm) and norm > 0, (name, norm)
            self._metrics['train'][f'adapter_grad_norm_{name}'].append(norm.item())
            norms.append(norm)
        return torch.stack(norms).norm()

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval'):
        started = time.perf_counter()
        previous, training = self.model.active_adapters[0], self.model.training
        metrics = {}
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            for name in self.task_names:
                self.model.set_adapter(name)
                self.model.eval()
                total, correct, count = 0., 0, 0
                for batch in DataLoader(eval_dataset or self.eval_dataset,
                                        batch_size=self.args.per_device_eval_batch_size,
                                        collate_fn=default_data_collator):
                    ids = batch['input_ids'].to('cuda')
                    logits = self.model(input_ids=ids, use_cache=False).logits[:, :-1].float()
                    target = ids[:, 1:]
                    total += F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                             target.reshape(-1), reduction='sum').item()
                    correct += (logits.argmax(-1) == target).sum().item()
                    count += target.numel()
                loss = total/count
                metrics.update({f'{metric_key_prefix}_{name}_loss': loss,
                                f'{metric_key_prefix}_{name}_perplexity': math.exp(loss),
                                f'{metric_key_prefix}_{name}_accuracy': correct/count})
        self.model.set_adapter(previous)
        self.model.train(training)
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self.eval_wall_seconds += time.perf_counter()-started
        return metrics


class SharedFrozenPrefix:
    """Training-pair-local reuse for this Qwen's frozen, task-independent prefix.

    Not a general KV cache. Evaluation/tracing always uses the original forwards.
    Requires no prefix adapters/pruning/dropout, identical inputs and no checkpointing.
    Stores one detached [B,S,hidden] tensor, never a dataset-sized activation cache.
    """
    def __init__(self, bridge):
        assert not bridge.original_model.config.use_cache
        assert not getattr(bridge.original_model.config, 'output_router_logits', False)
        self.enabled, self.cached, self.reused_calls = False, None, 0
        self.originals = []
        for index, block in enumerate(bridge.blocks[:min(LAYERS)]):
            assert not any(p.requires_grad for p in block.parameters())
            assert not any('lora_' in name for name, _ in block.named_modules())
            assert block.attn._original_component.attention_dropout == 0
            assert not any(isinstance(m, torch.nn.Dropout) and m.p > 0 for m in block.modules())
            original = block.forward
            self.originals.append((block, original))

            def forward(*args, original=original, last=index == min(LAYERS)-1, **kwargs):
                if not self.enabled:
                    return original(*args, **kwargs)
                if self.cached is not None:
                    self.reused_calls += 1
                    return self.cached if last else (args[0] if args else kwargs['hidden_states'])
                output = original(*args, **kwargs)
                if last:
                    assert isinstance(output, torch.Tensor) and not output.requires_grad
                    self.cached = output.detach().clone()
                return output

            object.__setattr__(block, 'forward', forward)

    def restore(self):
        for block, original in self.originals:
            object.__setattr__(block, 'forward', original)


def run(root, cfg, mode, prepared_dir=None):
    prepared_dir = root if prepared_dir is None else prepared_dir
    started = time.perf_counter()
    out = root/mode
    out.mkdir(exist_ok=False)
    names = list(RATIOS) if mode.startswith('joint') else [mode]
    set_seed(cfg.seed)
    model, tokenizer, revision = load_gptq_model()
    preparation = json.loads((prepared_dir/'preparation.json').read_text())
    if preparation['config'] != asdict(cfg):
        raise ValueError('Training config differs from the preparation config')
    assert revision == preparation['revision'], 'Checkpoint revision changed between runs'
    assert hashlib.sha256((prepared_dir/'data.pt').read_bytes()).hexdigest() == preparation['data_sha256']
    model = add_lora(model, names[0])
    for name in names[1:]:
        with patch('peft.utils.transformers_weight_conversion.convert_peft_config_for_transformers', return_value=None):
            model.add_adapter(name, copy.deepcopy(model.peft_config[names[0]]))
    bridge = wrap_qwen(model)
    params = {name: adapter_parameters(model, name) for name in names}
    initial = torch.load(prepared_dir/'initial_adapter.pt', weights_only=True)
    with torch.no_grad():
        for values in params.values():
            assert values.keys() == initial.keys()
            for key, p in values.items():
                p.copy_(initial[key])
    del initial
    data = torch.load(prepared_dir/'data.pt', weights_only=True)
    train, evaluation = Dataset.from_dict(data['train']), Dataset.from_dict(data['eval'])
    manifests = json.loads((prepared_dir/'plans.json').read_text())
    boundaries = {name: apply_plan(model, bridge, manifests[name], name) for name in names}
    TorchPruningGroupMapper.prepare_all_norms(bridge)
    tasks, description = make_tasks(bridge)
    probe = torch.tensor(evaluation[0]['input_ids'][:64], device='cuda')[None]
    mask_checks = {}
    model.eval()
    for name in names:
        model.set_adapter(name)
        mask_checks[name] = audit_masks(model, bridge, tasks[name], boundaries[name], probe[:, :16])

    hook_names = sorted({point.name for layer in LAYERS
                         for module in bridge.blocks[layer].modules() if isinstance(module, LinearBridge)
                         for point in (module.hook_in, module.hook_out)})

    def capture(adapter, phase):
        model.set_adapter(adapter)
        model.eval()
        hook_count = sum(len(p.fwd_hooks) for p in bridge.hook_dict.values())
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            logits = model(input_ids=probe, use_cache=False).logits.detach().cpu()
            trace = TransferPruningTracer.capture_hookpoints_activations_with_defects(
                bridge, probe, names_filter=hook_names, compute_gradient=False, groups=None).to_cpu()
        assert hook_count == sum(len(p.fwd_hooks) for p in bridge.hook_dict.values()), 'Capture removed SP hooks'
        # Distinguish the two masks: prune20 must NOT accidentally receive prune80's extra zeros.
        extra = sorted(set(description['prune80']['head_indices']) - set(description['prune20']['head_indices']))
        for layer in LAYERS:
            key = bridge.blocks[layer].attn.q.hook_out.name
            x = trace.corruptions[key].restore_activation(trace.activations[key])
            x = x.reshape(*x.shape[:-1], description[adapter]['query_heads'], description[adapter]['head_dim'])
            if adapter == 'prune20':
                assert x[..., extra].count_nonzero() > 0, 'prune80 mask leaked into prune20'
            else:
                assert x[..., extra].count_nonzero() == 0
        torch.save(trace, out/f'{adapter}_{phase}_trace.pt')
        torch.save(logits, out/f'{adapter}_{phase}_logits.pt')
        print('CAPTURE', adapter, phase, len(trace.activations), flush=True)

    for name in names:
        capture(name, 'before')
    gc.collect()
    torch.cuda.empty_cache()
    groups = {name: list(values.values()) for name, values in params.items()}
    adapter_ids = {id(p) for group in groups.values() for p in group}
    base = [p for p in model.parameters() if id(p) not in adapter_ids]
    assert all(not p.requires_grad for p in base)
    frozen_versions = [(t, t._version) for t in [*base, *model.buffers()]]
    optimizer = torch.optim.AdamW([{'params': values, 'adapter': name} for name,values in groups.items()],
                                  lr=cfg.learning_rate, weight_decay=.01, foreach=True)
    args = SFTConfig(
        output_dir=str(out/'trainer'), max_steps=cfg.steps,
        per_device_train_batch_size=cfg.batch_size, per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=1, gradient_checkpointing=False,
        learning_rate=cfg.learning_rate, warmup_steps=cfg.warmup_steps, lr_scheduler_type='cosine',
        max_grad_norm=1.0, fp16=True, bf16=False, logging_steps=1, disable_tqdm=True,
        eval_strategy='steps', eval_steps=cfg.eval_every, save_strategy='no', report_to='none',
        seed=cfg.seed, data_seed=cfg.seed, max_length=cfg.context_length, packing=False,
        remove_unused_columns=False, dataset_kwargs={'skip_prepare_dataset': True}, logging_nan_inf_filter=False,
    )
    metrics = TrainingMetrics(cfg, out)
    trainer = ExperimentTrainer(model=model, task_names=names, adapter_groups=groups,
        args=args, processing_class=tokenizer, train_dataset=train, eval_dataset=evaluation,
        data_collator=default_data_collator, optimizers=(optimizer, None), callbacks=[metrics])
    metrics.trainer = trainer
    if mode == 'joint_prefix':
        trainer.shared_prefix = SharedFrozenPrefix(bridge)
    skip_redundant_train(model)
    cache_parameter_iteration(model)
    cache_flop_count(trainer, {'input_ids': probe})
    before = trainer.evaluate()
    setup_seconds = time.perf_counter()-started
    set_seed(cfg.seed)
    print('TRAIN', mode, flush=True)
    train_started, eval_started = time.perf_counter(), trainer.eval_wall_seconds
    trainer.train()
    training_wall = time.perf_counter()-train_started
    history = [{'step': 0, **before}, *trainer.state.log_history]
    after = next(row for row in reversed(history) if f'eval_{names[0]}_loss' in row)
    assert not any(r['optimizer_step_skipped'] for r in metrics.steps), 'Overflow: independent equivalence is not guaranteed'
    assert all(t._version == version for t,version in frozen_versions), 'Frozen base/buffer was mutated'
    assert all(p.grad is None for p in base)
    parameter_steps, changed = {}, {}
    initial = torch.load(prepared_dir/'initial_adapter.pt', weights_only=True)
    for name in names:
        state = {key:p.detach().cpu().clone() for key,p in params[name].items()}
        changed[name] = sum(not torch.equal(value, initial[key]) for key,value in state.items())
        assert changed[name] > 0
        torch.save(state, out/f'{name}_adapter.pt')
        parameter_steps[name] = {key:int(optimizer.state[p].get('step', 0)) for key,p in params[name].items()}
        capture(name, 'after')
    steady = metrics.steps[2:]
    wall = sum(r['step_seconds'] for r in steady)
    result = dict(mode=mode, names=names, config=asdict(cfg), revision=revision,
        before=before, after=after, history=history, steps=metrics.steps,
        setup_seconds=setup_seconds, training_wall_seconds_including_eval=training_wall,
        scheduled_eval_seconds=trainer.eval_wall_seconds-eval_started,
        worker_total_seconds=time.perf_counter()-started,
        steady_pair_or_single_step_seconds=wall/len(steady),
        adapter_tokens_per_second=len(names)*cfg.batch_size*cfg.context_length*len(steady)/wall,
        peak_GiB=max(r['peak_allocated_GiB'] for r in metrics.steps),
        changed_tensors=changed, parameter_update_steps=parameter_steps,
        base_versions_unchanged=True, mask_checks=mask_checks,
        reuse_frozen_prefix=mode == 'joint_prefix',
        clipping='independent per adapter, max_norm=1; shared GradScaler, require no skipped steps')
    completed = out/'result.json'
    pending = completed.with_suffix('.tmp')
    pending.write_text(json.dumps(result, indent=2))
    pending.replace(completed)
    print('DONE', mode, before, after, 'STEP', result['steady_pair_or_single_step_seconds'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=['prepare', 'joint', 'joint_prefix', *RATIOS], required=True)
    parser.add_argument('--prepared-dir', type=Path, help='Reuse prepared masks, data and adapter initialization')
    args = parser.parse_args()
    cfg = ExperimentConfig(**json.loads((args.run_dir/'config.json').read_text()))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if args.mode == 'prepare':
        prepare(args.run_dir, cfg)
    else:
        run(args.run_dir, cfg, args.mode, args.prepared_dir)
