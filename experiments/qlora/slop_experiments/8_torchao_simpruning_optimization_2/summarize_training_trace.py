"""Summarize exported torch.profiler Chrome traces without loading the model."""
import json
from collections import defaultdict
from pathlib import Path


def summarize(path):
    events = json.loads(Path(path).read_text())['traceEvents']
    kernels = defaultdict(lambda: [0, 0.0])
    runtime = defaultdict(lambda: [0, 0.0])
    threads = defaultdict(list)
    phases = {}
    intervals = []
    for event in events:
        if event.get('ph') != 'X':
            continue
        category = event.get('cat', '')
        name, duration = event['name'], event['dur']
        if category == 'kernel':
            kernels[name][0] += 1
            kernels[name][1] += duration / 1e6
            intervals.append((event['ts'], event['ts'] + duration))
        elif category == 'cuda_runtime':
            runtime[name][0] += 1
            runtime[name][1] += duration / 1e6
        elif category in ('cpu_op', 'user_annotation'):
            threads[(event['pid'], event['tid'])].append(event)
        if name.startswith('phase.') and category == 'user_annotation':
            phases[name] = phases.get(name, 0.0) + duration / 1e6

    # Exclusive CPU event times: parent durations include nested operations.
    cpu_self = defaultdict(lambda: [0, 0.0])
    for values in threads.values():
        stack = []
        for event in sorted(values, key=lambda e: (e['ts'], -e['dur'])):
            while stack and event['ts'] >= stack[-1][0]:
                stack.pop()
            end = event['ts'] + event['dur']
            cpu_self[event['name']][0] += 1
            cpu_self[event['name']][1] += event['dur'] / 1e6
            if stack and end <= stack[-1][0]:
                cpu_self[stack[-1][1]][1] -= event['dur'] / 1e6
            stack.append((end, event['name']))

    busy = 0.0
    end = -float('inf')
    for start, finish in sorted(intervals):
        busy += max(0, finish - max(start, end))
        end = max(end, finish)

    def top(values, count=30):
        return [dict(name=name, count=row[0], seconds=row[1])
                for name, row in sorted(values.items(), key=lambda item: -item[1][1])[:count]]

    return dict(source=str(path), phases_seconds=phases,
                kernel_count=sum(row[0] for row in kernels.values()),
                kernel_sum_seconds=sum(row[1] for row in kernels.values()),
                kernel_busy_seconds=busy / 1e6,
                kernel_span_seconds=(max(b for a, b in intervals) - min(a for a, b in intervals)) / 1e6,
                top_kernels=top(kernels), cuda_runtime=top(runtime), cpu_self=top(cpu_self))


if __name__ == '__main__':
    root = Path(__file__).resolve().parent / 'torchao_trainer_overhead'
    result = {path.parent.name: summarize(path) for path in root.glob('*/step*.chrome.json')}
    (root / 'trace_summary.json').write_text(json.dumps(result, indent=2))
    for name, row in result.items():
        print(name, {key: value for key, value in row.items() if key not in
                     ('top_kernels', 'cuda_runtime', 'cpu_self')})
        print('KERNELS', row['top_kernels'][:5])
        print('CPU SELF', row['cpu_self'][:8])
