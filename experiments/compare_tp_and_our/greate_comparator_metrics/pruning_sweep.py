"""Reproducible, resumable SP/TP forward-only sweep; physical CUDA device 2 only.

Each model/dtype is loaded in a separate process. Within that process the original
Parameter objects and structural metadata are restored after EVERY configuration.
No optimizer, training, quantization, MLflow, or other GPU is used.
"""

import argparse
from dataclasses import asdict
import gc
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import warnings

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
MODELS = {
    'llama': 'meta-llama/Llama-3.1-8B-Instruct',
    'gemma': 'google/gemma-4-E4B-it',
    'deepseek': 'deepseek-ai/DeepSeek-V2-Lite-Chat',
}
CACHE = Path('/glazkov-dev/.cache')
DEFAULT_OUTPUT = Path(__file__).with_name('sweep_results')
COVERAGES = ('single', 'distributed', 'full')


def deepseek_nope_pruned_count(nope, ratio):
    """Keep the nearest positive even NOPE width; ties favor more pruning."""
    #see problem in https://github.com/TransformerLensOrg/TransformerLens/issues/1896
    retained = min(range(2, nope + 1, 2),
                   key=lambda size: (abs((nope - size) - nope * ratio), size))
    return nope - retained


def configurations(model, dtype, coverages=COVERAGES):
    for ratio, coverage, mlp in itertools.product(range(10, 100, 10), coverages, (False, True)):
        yield dict(id=f'{model}_{dtype}_{ratio:02d}_{coverage}_mlp{int(mlp)}',
                   model=model, dtype=dtype, ratio=ratio / 100, coverage=coverage,
                   mlp=mlp, seed=42, physical_gpu=2, batch_size=2, context_length=64,
                   compute_gradient=False, atol=1e-3, rtol=1e-2,
                   **({'nope_rounding': 'nearest_even_v1'} if model == 'deepseek' else {}))


def save_result(directory, row):
    # Runtime result artifacts, not source code; atomic and resumable.
    target = directory / (row['id'] + '.json')
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(row, indent=2, default=str, allow_nan=False))
    temporary.replace(target)


def tensor_stats(ref, test, atol=1e-3, rtol=1e-2):
    import torch
    from pruning_lens.core.activation_corruption import ActivationCorruption
    from pruning_lens.core.great_comparator import GreatComparator
    row = GreatComparator._compare_tensors(ref, test, ActivationCorruption([]),
                                           ActivationCorruption([]), atol, rtol, False)
    result = asdict(row)
    result.pop('difference')
    result['status'] = str(row.status)
    if row.max_abs is not None:
        delta = (test.float() - ref.float()).abs().flatten()
        result['p99_abs'] = delta.kthvalue(max(1, math.ceil(.99 * delta.numel()))).values.item()
        result['fraction_outside_tolerance'] = (
            (test.float() - ref.float()).abs() > atol + rtol * ref.float().abs()
        ).float().mean().item()
    return result


class ModelRestorePoint:
    """TP replaces Parameters, not their original values. Retain and restore owners.

Only used by this inference experiment (no optimizer or in-place weight updates).
Also restores the pre-hooks installed by structural setup and all shape metadata.
"""
    fields = ('in_features', 'out_features', 'head_dim', 'qk_head_dim',
              'qk_nope_head_dim', 'qk_rope_head_dim', 'scaling',
              '_qk_head_dim', '_qk_nope_head_dim', '_qk_rope_head_dim',
              '_mla_params_initialized')

    def __init__(self, model):
        self.model = model
        self.states = []
        for module in model.modules():
            self.states.append((module, dict(module._parameters),
                                {k: module.__dict__[k] for k in self.fields if k in module.__dict__},
                                module._forward_pre_hooks.copy(),
                                module._forward_pre_hooks_with_kwargs.copy()))

    def restore(self):
        self.model.reset_hooks()
        for module, parameters, fields, hooks, kwargs_hooks in self.states:
            module._parameters.clear()
            module._parameters.update(parameters)
            for key in self.fields:
                if key in fields:
                    module.__dict__[key] = fields[key]
                else:
                    module.__dict__.pop(key, None)
            module._forward_pre_hooks = hooks.copy()
            module._forward_pre_hooks_with_kwargs = kwargs_hooks.copy()
        for module, parameters, _, _, _ in self.states:
            assert all(module._parameters[k] is v for k, v in parameters.items())


def make_task(bridge, cfg):
    import torch
    from lora_transfer_pruning.core.prune_task_type import ModelPruneTask, GroupPruneTask
    from lora_transfer_pruning.adapter.torch_pruning.universal.gemma4_group_builder import Gemma4GroupBuilder
    n = len(bridge.blocks)
    layers = {'single': [0], 'distributed': [0, n // 2, n - 1], 'full': list(range(n))}[cfg['coverage']]
    tasks, effective_layers, mlp_layers = {}, set(), []
    for i in layers:
        # A shared-KV group already covers its provider and every consumer.
        # Use one root/mask per family, not independently sampled consumer masks.
        if cfg['coverage'] == 'full' and i in effective_layers:
            continue
        attn = bridge.blocks[i].attn
        hf = attn._original_component
        gen = torch.Generator().manual_seed(cfg['seed'] + i)
        if cfg['model'] == 'deepseek':
            nope, rope = hf.qk_nope_head_dim, hf.qk_rope_head_dim
            # KV latent width is unchanged; RoPE channels below are removed in pairs.
            count = deepseek_nope_pruned_count(nope, cfg['ratio'])
            a = torch.randperm(nope, generator=gen)[:count]
            b = torch.randperm(rope // 2, generator=gen)[:int(rope // 2 * cfg['ratio'])] * 2
            idxs = torch.cat((a, b + nope, b + 1 + nope)).sort().values.tolist()
            name = f'blocks.{i}.attn.q_proj'
        else:
            dim = hf.head_dim
            half = torch.randperm(dim // 2, generator=gen)[:int(dim // 2 * cfg['ratio'])]
            idxs = torch.cat((half, half + dim // 2)).sort().values.tolist()
            name = f'blocks.{i}.attn.q'
        tasks[name] = GroupPruneTask(cols=None, rows=idxs)
        effective_layers.update(Gemma4GroupBuilder.shared_kv_layers_family(bridge, i) if cfg['model'] == 'gemma' else [i])
    if cfg['mlp']:
        # DeepSeek's later layers are MoE; this sweep supports only its dense MLP 0.
        for i in ([0] if cfg['model'] == 'deepseek' else layers):
            module = bridge.get_submodule(f'blocks.{i}.mlp.up_proj')
            size = module._original_component.out_features
            gen = torch.Generator().manual_seed(cfg['seed'] + 1000 + i)
            idxs = torch.randperm(size, generator=gen)[:int(size * cfg['ratio'])].sort().values.tolist()
            tasks[f'blocks.{i}.mlp.up_proj'] = GroupPruneTask(cols=None, rows=idxs)
            mlp_layers.append(i)
    return ModelPruneTask(tasks, name=None), layers, sorted(effective_layers), mlp_layers


def capture(bridge, tokens, groups, tracer):
    import torch
    from pruning_lens.adapter.hook_names_filter import HookNamesFilter
    snapshots = {}

    def last_block_hook(module, args, output):
        output = output[0] if isinstance(output, tuple) else output
        snapshots['last_block'] = output.detach().cpu().clone()

    def logits_hook(module, args, output):
        snapshots['logits'] = output.detach().cpu().clone()

    handles = [bridge.blocks[-1].register_forward_hook(last_block_hook),
               bridge.register_forward_hook(logits_hook)]
    try:
        names = HookNamesFilter.filter(bridge, groups, None)
        trace = tracer.capture_hookpoints_activations_with_defects(
            bridge, tokens, names_filter=names, compute_gradient=False, groups=groups).to_cpu()
        return trace, snapshots
    finally:
        for handle in handles:
            handle.remove()


def run_configuration(bridge, tokens, cfg, baseline):
    import torch
    import torch.nn.functional as F
    from lora_transfer_pruning.usecase.local_pruning import LocalPruning
    from pruning_lens.adapter.transfer_pruning.transfer_pruning_tracer import TransferPruningTracer
    from pruning_lens.adapter.torch_pruning.torch_pruning_tracer import TorchPruningTracer
    from pruning_lens.core.great_comparator import GreatComparator
    torch.manual_seed(cfg['seed'])
    task, layers, affected, mlp_layers = make_task(bridge, cfg)
    local = LocalPruning(bridge, tokens[:1, :8])
    plan = local.get_torch_pruning_groups_and_structural_setups(task)
    removed = []
    for group in plan.groups:
        for dep, idxs in group:
            if isinstance(dep.target.module, torch.nn.Linear) and idxs:
                m = dep.target.module
                outgoing = group._DG.is_out_channel_pruning_fn(dep.handler)
                size = m.out_features if outgoing else m.in_features
                assert len(set(idxs)) == len(idxs) and min(idxs) >= 0 and max(idxs) < size, (
                    dep.target.name, 'out' if outgoing else 'in', size,
                    len(idxs), len(set(idxs)), min(idxs), max(idxs))
                removed.append(dict(module=dep.target.name, direction='out' if outgoing else 'in',
                                    original_size=size, removed=len(idxs), actual_ratio=len(idxs)/size,
                                    indices=list(idxs)))
    local.prepare_model_to_transfer_pruning_from_groups(plan.groups, rescale=False)
    sp, sp_outputs = capture(bridge, tokens, plan.groups, TransferPruningTracer)
    bridge.reset_hooks()
    for setup in plan.structural_setups:
        setup()
    for group in plan.groups:
        group.prune()
    for module in bridge.modules():
        if isinstance(module, torch.nn.Linear):
            assert tuple(module.weight.shape) == (module.out_features, module.in_features)
    tp, tp_outputs = capture(bridge, tokens, plan.groups, TorchPruningTracer)
    result = GreatComparator.compare_traces(tp, sp, atol=cfg['atol'], rtol=cfg['rtol'],
                                           compute_gradient=False, store_difference=False)
    hook_rows = {name: {k: v for k, v in asdict(row).items() if k != 'difference'}
                 for name, row in result.activations.items()}
    labels = tokens.cpu()[:, 1:].reshape(-1)

    def nll(logits):
        return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                               labels, reduction='none')

    tp_logits, sp_logits = tp_outputs['logits'], sp_outputs['logits']
    ref_nll, test_nll = nll(tp_logits), nll(sp_logits)
    log_p = tp_logits.float().log_softmax(-1)
    log_q = sp_logits.float().log_softmax(-1)
    kl = (log_p.exp() * (log_p - log_q)).sum(-1)
    values = {
        **cfg, 'status': 'completed', 'requested_layers': layers, 'effective_attn_layers': affected,
        'mlp_layers': mlp_layers, 'removed': removed, 'hooks': hook_rows,
        'hook_status_counts': {s: sum(str(row.status) == s for row in result.activations.values())
                               for s in sorted({str(row.status) for row in result.activations.values()})},
        'last_block': tensor_stats(tp_outputs['last_block'], sp_outputs['last_block'], cfg['atol'], cfg['rtol']),
        'logits': tensor_stats(tp_logits, sp_logits, cfg['atol'], cfg['rtol']),
        'metric_native': tensor_stats(tp.metric, sp.metric, cfg['atol'], cfg['rtol']),
        'metric_fp32': tensor_stats(ref_nll.mean(), test_nll.mean(), cfg['atol'], cfg['rtol']),
        'nll_tokens': tensor_stats(ref_nll, test_nll, cfg['atol'], cfg['rtol']),
        'loss_tp': ref_nll.mean().item(), 'loss_sp': test_nll.mean().item(),
        'loss_baseline': baseline['loss'],
        'kl_mean_tp_sp': kl.mean().item(), 'kl_max_tp_sp': kl.max().item(),
        'top1_disagreement': (tp_logits.argmax(-1) != sp_logits.argmax(-1)).float().mean().item(),
        'sp_rms_correction': 'existing dynamic zero-count coefficient (epsilon/rounding approximate)',
    }
    for name in ('tp', 'sp'):
        loss = values[f'loss_{name}']
        values[f'perplexity_{name}'] = math.exp(loss) if loss < 700 else None
    checked = [values[key]['status'] for key in ('last_block', 'logits', 'metric_native', 'metric_fp32')]
    checked.extend(values['hook_status_counts'])
    if any(s not in ('OK', 'DIFF') for s in checked):
        values['status'] = 'invalid_comparison'
    return values


def load_tokens(tokenizer, device):
    import torch
    import pyarrow.parquet as pq
    paths = sorted((CACHE / 'datasets--Salesforce--wikitext/snapshots').glob('*/wikitext-2-raw-v1/test-*.parquet'))
    if not paths:
        raise FileNotFoundError('Expected cached WikiText-2 raw TEST split')
    texts = pq.read_table(paths[0], columns=['text']).column('text').to_pylist()
    ids = []
    for text in texts:
        if text.strip():
            ids.extend(tokenizer(text, add_special_tokens=False).input_ids)
            ids.append(tokenizer.eos_token_id)
        if len(ids) >= 128:
            break
    tokens = torch.tensor(ids[:128], dtype=torch.long).reshape(2, 64).to(device)
    return tokens, str(paths[0])


def load_for_sweep(model_id, dtype, model_name):
    """Use cuda:2 only; untouched DeepSeek experts may execute on CPU if needed.

    No Accelerate meta/offload weights map: TP must own actual Parameters.
    Attention, dense MLP 0 and every pruned tensor stay on GPU. CPU expert
    placement is identical for SP and TP and explicitly recorded in the table.
    """
    import torch
    from transformers import AutoModelForCausalLM
    free = torch.cuda.mem_get_info(0)[0]
    if model_name == 'gemma' and dtype == torch.float32:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype,
                    device_map={'': 'cpu'}, attn_implementation='eager', local_files_only=True)
        # Gemma E4B's per-layer token table alone occupies about 10.5 GiB in FP32.
        # It is only looked up, never pruned by this grid. Keep its scale buffer
        # on CPU too, and transfer the looked-up activations back to the GPU.
        cpu_names = ['model.language_model.embed_tokens_per_layer',
                     'model.vision_tower', 'model.audio_tower']
        excluded_parameters, excluded_modules = set(), set()
        for name in cpu_names:
            component = model.get_submodule(name)
            excluded_parameters.update(id(p) for p in component.parameters())
            excluded_modules.update(id(m) for m in component.modules())
        table = model.get_submodule(cpu_names[0])
        table.register_forward_pre_hook(lambda module, args: (args[0].to('cpu'),))
        table.register_forward_hook(lambda module, args, output: output.to('cuda:0'))
        required = sum(p.numel() * p.element_size() for p in model.parameters()
                       if id(p) not in excluded_parameters)
        if required > free - 4 * 2**30:
            raise MemoryError('Insufficient cuda:2 memory for Gemma FP32 even with the per-layer table on CPU')
        for p in model.parameters():
            if id(p) not in excluded_parameters:
                p.data = p.data.to('cuda:0')
        for module in model.modules():
            if id(module) not in excluded_modules:
                for name, buffer in module._buffers.items():
                    if buffer is not None:
                        module._buffers[name] = buffer.to('cuda:0')
        if hasattr(model, 'hf_device_map'):
            del model.hf_device_map
        model._sweep_cpu_modules = cpu_names
        return model, []
    if model_name != 'deepseek':
        return AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype,
                    device_map={'': 'cuda:0'}, attn_implementation='eager', local_files_only=True), []
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype,
                device_map={'': 'cpu'}, attn_implementation='eager', local_files_only=True)
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    budget = free - 4 * 2**30
    cpu_layers, cpu_parameters = [], set()
    for i, layer in enumerate(model.model.layers):
        experts = getattr(layer.mlp, 'experts', None)
        if total <= budget:
            break
        if experts is not None:
            cpu_layers.append(i)
            for p in experts.parameters():
                cpu_parameters.add(id(p))
                total -= p.numel() * p.element_size()

            def to_cpu(module, args):
                return tuple(t.to('cpu') if isinstance(t, torch.Tensor) else t for t in args)

            def to_gpu(module, args, output):
                return output.to('cuda:0')

            experts.register_forward_pre_hook(to_cpu)
            experts.register_forward_hook(to_gpu)
    if total > budget:
        raise MemoryError('Insufficient cuda:2 memory even with all routed experts on CPU')
    for p in model.parameters():
        if id(p) not in cpu_parameters:
            p.data = p.data.to('cuda:0')
    for module in model.modules():
        for name, buffer in module._buffers.items():
            if buffer is not None:
                module._buffers[name] = buffer.to('cuda:0')
    # This is explicit tensor-transfer execution, NOT Accelerate offload. Its
    # load-time {'': 'cpu'} map is obsolete; retaining it would misdescribe where
    # the actual Parameters live. Placement is recorded separately in results.
    if hasattr(model, 'hf_device_map'):
        del model.hf_device_map
    model._sweep_cpu_modules = [f'model.layers.{i}.mlp.experts' for i in cpu_layers]
    return model, cpu_layers


def worker(model_name, dtype_name, output, limit=None, coverages=COVERAGES):
    # Must precede torch/transformers imports. Logical cuda:0 IS physical cuda:2.
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    os.environ['HF_HUB_CACHE'] = str(CACHE)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformer_lens.model_bridge import TransformerBridge
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    warnings.filterwarnings('ignore', message='Unwrapped parameters detected:')
    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[dtype_name]
    def needs_run(cfg):
        path = output / (cfg['id'] + '.json')
        if not path.exists():
            return True
        previous = json.loads(path.read_text())
        return (previous['status'] != 'completed'
                or previous.get('nope_rounding') != cfg.get('nope_rounding'))
    pending = [c for c in configurations(model_name, dtype_name, coverages) if needs_run(c)]
    if limit:
        pending = pending[:limit]
    if not pending:
        return
    model_id = MODELS[model_name]
    try:
        print(f'LOAD {model_id} {dtype_name}', flush=True)
        # FP32 whole Llama/Gemma needs about 30 GiB plus tracing workspace.
        # Do not intentionally OOM a shared GPU just to mark an unsupported run.
        if dtype_name == 'fp32' and model_name == 'llama' and torch.cuda.mem_get_info(0)[0] < 35 * 2**30:
            for cfg in pending:
                save_result(output, {**cfg, 'status': 'skipped_memory',
                                     'reason': 'FP32 requires ~35 GiB free on physical cuda:2; no other GPU allowed'})
            return
        model, cpu_expert_layers = load_for_sweep(model_id, dtype, model_name)
        cpu_modules = getattr(model, '_sweep_cpu_modules', [])
        bridge = TransformerBridge.boot_transformers(model_id, hf_model=model, dtype=dtype).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        tokens, data_path = load_tokens(tokenizer, 'cuda:0')
        with torch.no_grad():
            logits = bridge(input=tokens, return_type='logits', use_cache=False).detach().cpu()
        baseline = {'loss': F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                                           tokens.cpu()[:, 1:].reshape(-1)).item()}
        restore = ModelRestorePoint(bridge)
        manifest = dict(model_id=model_id, dtype=dtype_name, data_path=data_path,
                        tokens=tokens.cpu().tolist(), token_sha256=hashlib.sha256(tokens.cpu().numpy().tobytes()).hexdigest(),
                        torch=torch.__version__, device=torch.cuda.get_device_name(0),
                        baseline_loss=baseline['loss'], baseline_perplexity=math.exp(baseline['loss']),
                        parameters=sum(p.numel() for p in bridge.parameters()),
                        cpu_expert_layers=cpu_expert_layers, cpu_modules=cpu_modules)
        (output / f'manifest_{model_name}_{dtype_name}.json').write_text(json.dumps(manifest, indent=2))
    except Exception as exc:
        error = traceback.format_exc()
        print(error, flush=True)
        for cfg in pending:
            save_result(output, {**cfg, 'status': 'load_error', 'error': error})
        return
    for cfg in pending:
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        try:
            row = run_configuration(bridge, tokens, cfg, baseline)
        except Exception:
            row = {**cfg, 'status': 'error', 'error': traceback.format_exc()}
            print(row['error'], flush=True)
        finally:
            restore.restore()
            gc.collect()
            torch.cuda.empty_cache()
        row.update(seconds=time.monotonic() - started,
                   cpu_expert_layers=cpu_expert_layers,
                   cpu_modules=cpu_modules,
                   peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
        # JSON must not silently serialize NaN/Infinity as apparently valid numbers.
        def finite(value):
            if isinstance(value, dict): return {k: finite(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)): return [finite(v) for v in value]
            if isinstance(value, float) and not math.isfinite(value): return None
            return value
        save_result(output, finite(row))
        print(cfg['id'], row['status'], row.get('metric_fp32', {}), flush=True)
    with torch.no_grad():
        restored_logits = bridge(input=tokens, return_type='logits', use_cache=False).detach().cpu()
    audit = tensor_stats(logits, restored_logits, 0., 0.)
    (output / f'restore_audit_{model_name}_{dtype_name}.json').write_text(json.dumps(audit, indent=2))
    assert audit['status'] == 'OK', f'Original model restoration failed: {audit}'


def summary(output):
    import pandas as pd
    rows = [json.loads(p.read_text()) for p in output.glob('*.json') if not p.name.startswith(('manifest_', 'restore_audit_'))]
    for row in rows:
        row.setdefault('cpu_modules', [f'model.layers.{i}.mlp.experts' for i in row.get('cpu_expert_layers', [])])
    compact = [{k: v for k, v in row.items() if k not in ('hooks', 'removed', 'error')} for row in rows]
    frame = pd.json_normalize(compact)
    frame.to_csv(output / 'summary.csv', index=False)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=MODELS)
    parser.add_argument('--dtype', default='bf16', choices=('bf16', 'fp16', 'fp32'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--models', nargs='+', default=list(MODELS), choices=MODELS)
    parser.add_argument('--dtypes', nargs='+', default=['bf16', 'fp16', 'fp32'])
    parser.add_argument('--coverages', nargs='+', default=list(COVERAGES), choices=COVERAGES)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args.worker, args.dtype, args.output, args.limit, args.coverages)
        return
    for model, dtype in itertools.product(args.models, args.dtypes):
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', model,
                   '--dtype', dtype, '--output', str(args.output), '--coverages', *args.coverages]
        if args.limit:
            command.extend(['--limit', str(args.limit)])
        with (args.output / f'{model}_{dtype}.log').open('a') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        print(model, dtype, 'exit', result.returncode, flush=True)
        summary(args.output)


if __name__ == '__main__':
    main()
