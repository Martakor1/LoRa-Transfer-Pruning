import torch


from dataclasses import dataclass
from lora_transfer_pruning.core.prune_task_type import GroupPruneTask


@dataclass
class TorchPruningGroupTask:
    """
    A dataclass representing a pruning task for one torch pruning group.

    Attributes:
        cols (torch.Tensor | float | None): The indices of the columns to be pruned, or a float representing the pruning ratio, or None if no pruning is to be done.
        rows (torch.Tensor | float | None): The indices of the rows to be pruned, or a float representing the pruning ratio, or None if no pruning is to be done.
        kv_lora_idxs_deepseek (torch.Tensor | float | None): The indices of the kv_a_proj_with_mqa parameters for kv_lora_rank part inside to be pruned, or a float representing the pruning ratio, or None if no pruning is to be done.
    """
    cols: torch.Tensor | float | None
    rows: torch.Tensor | float | None
    kv_lora_idxs_deepseek: torch.Tensor | float | None = None

    @staticmethod
    def from_group_prune_task(task: GroupPruneTask, device: torch.device) -> "TorchPruningGroupTask":
        def to_tensor_or_float(x):
            if (isinstance(x, list)):
                return torch.tensor(x, dtype=torch.long, device=device)
            return x
        return TorchPruningGroupTask(
            cols=to_tensor_or_float(task.cols),
            rows=to_tensor_or_float(task.rows),
            kv_lora_idxs_deepseek=to_tensor_or_float(task.kv_lora_idxs_deepseek),
        )