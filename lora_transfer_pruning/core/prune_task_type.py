from dataclasses import dataclass


@dataclass
class GroupPruneTask:
    """
    A dataclass representing a pruning task for one group.

    Attributes:
        cols (list[int] | float | None): The indices of the columns to be pruned, or a float representing the pruning ratio, or None if no pruning is to be done.
        rows (list[int] | float | None): The indices of the rows to be pruned, or a float representing the pruning ratio, or None if no pruning is to be done.
        kv_lora_idxs_deepseek (list[int] | float | None): The indices of the kv_a_proj_with_mqa parameters for kv_lora_rank part inside to be pruned, or a float representing the pruning ratio, or None if no pruning is to be done.
    """
    cols: list[int] | float | None
    rows: list[int] | float | None
    kv_lora_idxs_deepseek: list[int] | float | None = None

ModelPruneTask = dict[str, GroupPruneTask]