"""CPU-only comparisons of trusted local experiment artifacts."""
import argparse
import json
import sys
from pathlib import Path
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import torch
import pandas as pd
from pruning_lens.core.great_comparator import GreatComparator


def compare(root):
    torch.set_num_threads(4)
    results = {name: json.loads((root/name/'result.json').read_text())
               for name in ('joint', 'joint_prefix', 'prune20', 'prune80')}
    weights, traces, checks, logits_rows = [], [], [], []
    for candidate_mode in ('joint', 'joint_prefix'):
        for adapter in ('prune20', 'prune80'):
            reference = torch.load(root/adapter/f'{adapter}_adapter.pt', weights_only=True)
            candidate = torch.load(root/candidate_mode/f'{adapter}_adapter.pt', weights_only=True)
            assert reference.keys() == candidate.keys()
            for name, ref in reference.items():
                test = candidate[name]
                delta = test.double() - ref.double()
                weights.append(dict(candidate=candidate_mode, adapter=adapter, parameter=name,
                    exact=torch.equal(ref, test), max_abs=delta.abs().max().item(),
                    rmse=delta.square().mean().sqrt().item(),
                    rel_l2=delta.norm().item()/max(ref.double().norm().item(), 1e-12)))
            del reference, candidate
            r, c = results[adapter], results[candidate_mode]
            def history(result, key):
                return [row[key] for row in result['history'] if key in row]
            losses_r, losses_c = history(r, f'adapter_loss_{adapter}'), history(c, f'adapter_loss_{adapter}')
            assert len(losses_r) == len(losses_c) == r['config']['steps']
            checks.append(dict(candidate=candidate_mode, adapter=adapter,
                losses_exact=losses_r == losses_c,
                loss_max_abs=max(abs(a-b) for a,b in zip(losses_r, losses_c)),
                grad_norms_exact=history(r, f'adapter_grad_norm_{adapter}') == history(c, f'adapter_grad_norm_{adapter}'),
                update_counts_exact=r['parameter_update_steps'][adapter] == c['parameter_update_steps'][adapter]))
            for phase in ('before', 'after'):
                # These are our own locally-created dataclass artifacts, not untrusted downloads.
                ref = torch.load(root/adapter/f'{adapter}_{phase}_trace.pt', weights_only=False)
                test = torch.load(root/candidate_mode/f'{adapter}_{phase}_trace.pt', weights_only=False)
                comparison = GreatComparator.compare_traces(ref, test, atol=1e-4, rtol=1e-3,
                                                            compute_gradient=False, store_difference=False)
                assert comparison.activations
                for name, row in comparison.activations.items():
                    traces.append(dict(candidate=candidate_mode, adapter=adapter, phase=phase,
                        hook=name, status=str(row.status), max_abs=row.max_abs, rel_l2=row.rel_l2, rmse=row.rmse))
                ref_logits = torch.load(root/adapter/f'{adapter}_{phase}_logits.pt', weights_only=True).float()
                test_logits = torch.load(root/candidate_mode/f'{adapter}_{phase}_logits.pt', weights_only=True).float()
                delta = test_logits-ref_logits
                logits_rows.append(dict(candidate=candidate_mode, adapter=adapter, phase=phase,
                    exact=torch.equal(ref_logits, test_logits), max_abs=delta.abs().max().item(),
                    rel_l2=delta.norm().item()/max(ref_logits.norm().item(),1e-12),
                    top1_disagreement=(ref_logits.argmax(-1) != test_logits.argmax(-1)).float().mean().item()))
                # Non-vacuous comparator check: a deliberately damaged activation must fail.
                key = next(iter(test.activations))
                damaged = dict(test.activations)
                damaged[key] = damaged[key] + 1
                negative = GreatComparator.compare_traces(test, replace(test, activations=damaged),
                    compute_gradient=False, store_difference=False)
                assert str(negative.activations[key].status) == 'DIFF'
    for name, rows in (('weights', weights), ('traces', traces), ('checks', checks), ('logits', logits_rows)):
        pd.DataFrame(rows).to_csv(root/f'comparison_{name}.csv', index=False)
    summary = dict(
        checks=checks, all_adapter_tensors_exact=all(row['exact'] for row in weights),
        worst_weight_max_abs=max(row['max_abs'] for row in weights),
        trace_status_counts=pd.Series(row['status'] for row in traces).value_counts().to_dict(),
        all_logits_exact=all(row['exact'] for row in logits_rows), negative_control_passed=True,
    )
    completed = root/'comparison.json'
    pending = completed.with_suffix('.tmp')
    pending.write_text(json.dumps(summary, indent=2))
    pending.replace(completed)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    compare(parser.parse_args().run_dir)
