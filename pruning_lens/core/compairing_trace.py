from dataclasses import dataclass, field
from enum import Enum

import torch


class ComparisonStatus(str, Enum):
    OK = "OK"
    DIFF = "DIFF"
    MISSING = "MISSING"
    NO_GRAD = "NO_GRAD"
    SHAPE = "SHAPE"
    UNRESOLVED = "UNRESOLVED"
    NONFINITE = "NONFINITE"
    RESTORE_ERROR = "RESTORE_ERROR"
    EMPTY = "EMPTY"

    def __str__(self) -> str:
        return self.value


@dataclass
class TensorComparison:
    '''Restored comparison; difference is candidate - reference on CPU.

    difference covers the full restored shape and is omitted when store_difference=False.
    '''
    status: ComparisonStatus
    difference: torch.Tensor | None = None
    max_abs: float | None = None
    mean_abs: float | None = None
    rmse: float | None = None
    rel_l2: float | None = None
    abs_l2: float | None = None
    reference_shape: tuple[int, ...] | None = None
    candidate_shape: tuple[int, ...] | None = None
    reference_restored_shape: tuple[int, ...] | None = None
    candidate_restored_shape: tuple[int, ...] | None = None
    compared_elements: int = 0
    reason: str | None = None

@dataclass
class CompairingTrace:
    '''Per-hook comparisons. Reference=TP, candidate=SP in the comparison usecase.

    An empty gradients dictionary means backward comparison was not requested.
    '''
    activations: dict[str, TensorComparison] = field(default_factory=dict)
    gradients: dict[str, TensorComparison] = field(default_factory=dict)
    metric: TensorComparison | None = None
    atol: float = 1e-4
    rtol: float = 1e-3

    def print_summary(self) -> None:
        '''Print stored statistics; first difference follows trace insertion order.'''
        names = list(dict.fromkeys([*self.activations, *self.gradients]))
        for index, name in enumerate(names):
            print(f"[{index:02d}] {name}")
            for label, rows in (("FWD", self.activations), ("BWD", self.gradients)):
                if name in rows:
                    self._print_comparison(label, rows[name])
        if self.metric is not None:
            self._print_comparison("METRIC", self.metric)
        for label, rows in (("FWD", self.activations), ("BWD", self.gradients)):
            if not rows:
                continue
            first = next((name for name, row in rows.items() if row.status == ComparisonStatus.DIFF), None)
            incomplete = sum(row.status not in (ComparisonStatus.OK, ComparisonStatus.DIFF) for row in rows.values())
            print(f"FIRST {label} DIFF: {first}; not compared successfully: {incomplete}")

    @staticmethod
    def _print_comparison(label: str, row: TensorComparison) -> None:
        detail = (
            f"shape={row.reference_restored_shape} "
            f"compared={row.compared_elements}"
        )
        if row.max_abs is not None:
            detail += (
                f" max_abs={row.max_abs:.3e} mean_abs={row.mean_abs:.3e}"
                f" rmse={row.rmse:.3e} rel_l2={row.rel_l2:.3e} abs_l2={row.abs_l2:.3e}"
            )
        if row.status == ComparisonStatus.SHAPE:
            detail += f" candidate_shape={row.candidate_restored_shape}"
        if row.reason:
            detail += f" ({row.reason})"
        print(f"     {label:6s} {row.status:13s} {detail}")
