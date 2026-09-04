from dataclasses import dataclass, field


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

@dataclass
class ModelPruneTask:
    """
    A dataclass representing a pruning task for a model.

    Attributes:
        name (str): The name of prune task. Should be the same as the name of the LoRA adapter, which is "default" by default.
        data (dict[str, GroupPruneTask]): A dictionary mapping full module names to their corresponding pruning tasks.
    """
     #due to the fact, that LoRA adapter defaul name is "default"
    data: dict[str, GroupPruneTask]
    name: str = "default"
