import torch

from .activation_corruption import ActivationCorruption
from .compairing_trace import CompairingTrace, ComparisonStatus, TensorComparison
from .pruning_trace import PruningTrace

class GreatComparator:

    @staticmethod
    def compare_traces(
        reference_trace: PruningTrace,
        candidate_trace: PruningTrace,
        atol: float = 1e-4,
        rtol: float = 1e-3,
        *,
        compute_gradient: bool = True,
        store_difference: bool = True,
    ) -> CompairingTrace:
        '''Restore through each trace's defects and compare on CPU.

        Activations and gradients include all restored coordinates.
        Unknown mappings never receive OK. No architectural alignment heuristics
        or model mutation are performed here.
        '''
        result = CompairingTrace(atol=atol, rtol=rtol)
        names = list(dict.fromkeys([*reference_trace.activations, *candidate_trace.activations]))
        for name in names:
            result.activations[name] = GreatComparator._compare_hook(
                reference_trace, candidate_trace, name, False,
                atol, rtol, store_difference,
            )
            if compute_gradient:
                result.gradients[name] = GreatComparator._compare_hook(
                    reference_trace, candidate_trace, name, True,
                    atol, rtol, store_difference,
                )
        result.metric = GreatComparator._compare_tensors(
            reference_trace.metric, candidate_trace.metric,
            ActivationCorruption([]), ActivationCorruption([]),
            atol, rtol, store_difference,
        )
        return result

    @staticmethod
    def _compare_hook(reference, candidate, name, backward, atol, rtol, store_difference):
        if name not in reference.activations or name not in candidate.activations:
            return TensorComparison(
                status=ComparisonStatus.MISSING,
                reason=f"reference={name in reference.activations}, candidate={name in candidate.activations}",
            )
        ref = (reference.gradients if backward else reference.activations).get(name)
        test = (candidate.gradients if backward else candidate.activations).get(name)
        shapes = dict(
            reference_shape=None if ref is None else tuple(ref.shape),
            candidate_shape=None if test is None else tuple(test.shape),
        )
        if backward and (ref is None or test is None):
            return TensorComparison(
                status=ComparisonStatus.NO_GRAD, **shapes,
                reason=f"reference gradient={ref is not None}, candidate gradient={test is not None}",
            )
        if (name in reference.unresolved_pruning_hooks
                or name in candidate.unresolved_pruning_hooks
                or name not in reference.corruptions or name not in candidate.corruptions):
            return TensorComparison(status=ComparisonStatus.UNRESOLVED, **shapes, reason="Pruning/layout mapping is unknown")
        return GreatComparator._compare_tensors(
            ref, test, reference.corruptions[name], candidate.corruptions[name],
            atol, rtol, store_difference,
        )

    @staticmethod
    def _compare_tensors(ref, test, ref_corruption, test_corruption, atol, rtol, store_difference):
        row = TensorComparison(
            status=ComparisonStatus.OK, reference_shape=tuple(ref.shape), candidate_shape=tuple(test.shape),
        )
        # Convert before subtraction: BF16 subtraction would round the difference.
        dtype = torch.float64 if torch.float64 in (ref.dtype, test.dtype) else torch.float32
        ref = ref.detach().to(device="cpu", dtype=dtype)
        test = test.detach().to(device="cpu", dtype=dtype)
        try:
            restored_ref = ref_corruption.restore_activation(ref)
            restored_test = test_corruption.restore_activation(test)
            row.reference_restored_shape = tuple(restored_ref.shape)
            row.candidate_restored_shape = tuple(restored_test.shape)
            if restored_ref.shape != restored_test.shape:
                row.status = ComparisonStatus.SHAPE
                return row
        except (ValueError, RuntimeError, IndexError, TypeError) as exc:
            row.status, row.reason = ComparisonStatus.RESTORE_ERROR, str(exc)
            return row

        difference = restored_test - restored_ref
        if store_difference:
            row.difference = difference
        row.compared_elements = restored_ref.numel()
        if not row.compared_elements:
            row.status, row.reason = ComparisonStatus.EMPTY, "No comparable coordinates"
            return row
        if not all(torch.isfinite(tensor).all().item() for tensor in (restored_ref, restored_test, difference)):
            row.status, row.reason = ComparisonStatus.NONFINITE, "NaN/Inf in compared coordinates or their difference"
            return row

        delta64 = difference.double()  # Avoid overflow of FP32 squared norms.
        row.max_abs = delta64.abs().max().item()
        row.mean_abs = delta64.abs().mean().item()
        row.rmse = delta64.square().mean().sqrt().item()
        row.abs_l2 = delta64.norm().item()
        row.rel_l2 = row.abs_l2 / max(restored_ref.double().norm().item(), 1e-12)
        row.status = ComparisonStatus.OK if torch.allclose(restored_test, restored_ref, atol=atol, rtol=rtol) else ComparisonStatus.DIFF
        return row
        
        
