"""Run five paired Qwen INT4 timing repeats on physical GPU 2."""
import argparse
import csv
import fcntl
import itertools
import json
import math
import os
import random
import statistics
import subprocess
import sys
from tempfile import mkdtemp
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SERIAL = ('prune20', 'prune80')
MODES = ('serial', 'joint', 'joint_prefix')
GPU_INDEX = 2
ORDER_SEED = 20261118


def run_repeats(output, prepared):
    files = ('config.json', 'preparation.json', 'plans.json', 'data.pt', 'initial_adapter.pt')
    for name in files:
        if not (prepared / name).is_file():
            raise FileNotFoundError(prepared / name)
    config = json.loads((prepared / 'config.json').read_text())
    preparation = json.loads((prepared / 'preparation.json').read_text())
    if config != preparation['config']:
        raise ValueError('Training config differs from the preparation config')

    gpu = subprocess.check_output(
        ['nvidia-smi', '-i', str(GPU_INDEX), '--query-gpu=uuid,name,driver_version',
         '--format=csv,noheader'], text=True,
    ).strip()
    gpu_uuid, gpu_name, driver = gpu.split(', ')
    order = list(itertools.permutations(MODES))
    random.Random(ORDER_SEED).shuffle(order)
    worker_orders = []
    for index, block in enumerate(order[:5]):
        singles = SERIAL if index % 2 == 0 else SERIAL[::-1]
        worker_orders.append([worker for mode in block for worker in (singles if mode == 'serial' else (mode,))])
    protocol = dict(gpu_index=GPU_INDEX, gpu_uuid=gpu_uuid, gpu_name=gpu_name, driver=driver,
                    order_seed=ORDER_SEED, worker_orders=worker_orders,
                    prepared_dir=os.path.relpath(prepared, output), config=config,
                    revision=preparation['revision'], data_sha256=preparation['data_sha256'])
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError(f'Run protocol changed: {protocol_path}')
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2) + '\n')

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid, COMPUTE_DTYPE='fp16',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', TOKENIZERS_PARALLELISM='false')
    for index, workers in enumerate(worker_orders, 1):
        block = output / f'run_{index:02}'
        block.mkdir(exist_ok=True)
        (block / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        for mode in workers:
            arm = block / mode
            if (arm / 'result.json').exists():
                continue
            log_path = block / f'{mode}.log'
            if arm.exists() or log_path.exists():
                failed = block / 'incomplete'
                failed.mkdir(exist_ok=True)
                archived = Path(mkdtemp(prefix=f'{mode}-', dir=failed))
                for path in (arm, log_path):
                    if path.exists():
                        path.rename(archived / path.name)
                print(f'Preserved incomplete attempt in {archived}', flush=True)
            print(f'Run {index}/5: {mode}', flush=True)
            with log_path.open('a') as log:
                subprocess.run([sys.executable, '-u', str(HERE / 'qwen_torchao_multiadapter.py'),
                                '--run-dir', str(block), '--prepared-dir', str(prepared), '--mode', mode],
                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        if not (block / 'comparison.json').exists():
            print(f'Run {index}/5: checking adapter equivalence', flush=True)
            with (block / 'comparison.log').open('a') as log:
                subprocess.run([sys.executable, str(HERE / 'compare_qwen_multiadapter.py'), str(block)],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    summarize(output)
    print('Five repeats complete', flush=True)


TIME_FIELDS = {
    'pair_step_s': 'steady_pair_or_single_step_seconds',
    'train_with_eval_s': 'training_wall_seconds_including_eval',
    'worker_total_s': 'worker_total_seconds',
}
METRICS = (*TIME_FIELDS, 'adapter_tokens_s', 'peak_GiB')
CONTRASTS = (('serial', 'joint'), ('serial', 'joint_prefix'), ('joint', 'joint_prefix'))


def read_block(block, config, revision):
    comparison = json.loads((block / 'comparison.json').read_text())
    if not (comparison['all_adapter_tensors_exact'] and comparison['all_logits_exact']
            and comparison['negative_control_passed']
            and set(comparison['trace_status_counts']) == {'OK'}
            and all(check['losses_exact'] and check['grad_norms_exact'] and check['update_counts_exact']
                    for check in comparison['checks'])):
        raise ValueError(f'Adapter equivalence checks failed: {block}')

    workers = {}
    for mode in (*SERIAL, 'joint', 'joint_prefix'):
        result = json.loads((block / mode / 'result.json').read_text())
        if (result['config'] != config or result['revision'] != revision or result['mode'] != mode
                or len(result['steps']) != config['steps']
                or any(step['optimizer_step_skipped'] for step in result['steps'])
                or not result['base_versions_unchanged']):
            raise ValueError(f'Invalid timing result: {block / mode}')
        workers[mode] = result

    tokens_per_pair = 2 * config['batch_size'] * config['context_length']
    rows = []
    for mode in MODES:
        arms = [workers[name] for name in (SERIAL if mode == 'serial' else (mode,))]
        times = {metric: sum(arm[field] for arm in arms) for metric, field in TIME_FIELDS.items()}
        if not all(math.isfinite(t) and t > 0 for t in times.values()):
            raise ValueError(f'Invalid times: {block / mode}')
        rows.append(dict(run=block.name, mode=mode, **times,
                         adapter_tokens_s=tokens_per_pair / times['pair_step_s'],
                         peak_GiB=max(arm['peak_GiB'] for arm in arms)))
    return rows


def paired_comparisons(rows, metric):
    from scipy.stats import t

    times = {mode: [row[metric] for row in rows if row['mode'] == mode] for mode in MODES}
    comparisons = []
    for reference, candidate in CONTRASTS:
        log_ratios = [math.log(a / b) for a, b in zip(times[reference], times[candidate])]
        n = len(log_ratios)
        mean = statistics.mean(log_ratios)
        se = statistics.stdev(log_ratios) / math.sqrt(n)
        margin = t.ppf(0.975, df=n - 1) * se
        if se:
            p = 2 * t.sf(abs(mean / se), df=n - 1)
        else:
            p = 1.0 if mean == 0 else 0.0
        observed = abs(sum(log_ratios))
        flips = [abs(sum(sign * value for sign, value in zip(signs, log_ratios)))
                 for signs in itertools.product((-1, 1), repeat=n)]
        exact_p = sum(value >= observed or math.isclose(value, observed, rel_tol=1e-12)
                      for value in flips) / len(flips)
        comparisons.append(dict(metric=metric, reference=reference, candidate=candidate, n=n,
                                speedup=math.exp(mean), ci95_low=math.exp(mean - margin),
                                ci95_high=math.exp(mean + margin), paired_t_p=p, sign_flip_p=exact_p))
    return comparisons


def adjust_holm(comparisons):
    adjusted = 0.0
    for rank, row in enumerate(sorted(comparisons, key=lambda row: row['paired_t_p'])):
        adjusted = max(adjusted, min(1.0, (len(comparisons) - rank) * row['paired_t_p']))
        row['paired_t_p_holm'] = adjusted

def write_csv(path, rows):
    pending = path.with_suffix('.tmp')
    with pending.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    pending.replace(path)


def summarize(output):
    protocol = json.loads((output / 'protocol.json').read_text())
    blocks = [output / f'run_{index:02}' for index in range(1, len(protocol['worker_orders']) + 1)]
    rows = [row for block in blocks for row in read_block(block, protocol['config'], protocol['revision'])]
    summary = []
    for mode in MODES:
        mode_rows = [row for row in rows if row['mode'] == mode]
        for metric in METRICS:
            values = [row[metric] for row in mode_rows]
            summary.append(dict(mode=mode, metric=metric, n=len(values),
                                mean=statistics.mean(values), sd=statistics.stdev(values)))
    primary = paired_comparisons(rows, 'pair_step_s')
    secondary = [row for metric in ('train_with_eval_s', 'worker_total_s')
                 for row in paired_comparisons(rows, metric)]
    adjust_holm(primary)
    adjust_holm(secondary)
    write_csv(output / 'timings.csv', rows)
    write_csv(output / 'timing_summary.csv', summary)
    write_csv(output / 'paired_comparisons.csv', primary + secondary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--prepared-dir', type=Path, default=HERE / 'torchao_multiadapter_20261112')
    parser.add_argument('--summary-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.summary_only:
            summarize(output)
        else:
            run_repeats(output, args.prepared_dir.resolve())
