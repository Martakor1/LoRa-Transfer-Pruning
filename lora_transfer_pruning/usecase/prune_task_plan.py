from dataclasses import dataclass
from typing import Callable
import torch_pruning as tp

@dataclass
class PruneTaskPlan:
    '''A plan for pruning a model, including the groups to prune and any structural setups needed.
    
    Attributes:
        name (str | None): The name of the pruning task. If no name provided, preparation methods will add hooks to pruned modules without bounding to model's active adapters 
        groups (list[tp.Group]): A list of groups to be pruned.
        structural_setups (list[Callable[[], None]]): A list of callables that change model's inner constants like .head_dim (need for torch_pruning only)
    '''
    name: str | None
    groups: list[tp.Group]
    structural_setups: list[Callable[[], None]]