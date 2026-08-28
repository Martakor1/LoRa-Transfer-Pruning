from typing import Callable

import torch

from transformer_lens.model_bridge.bridge import TransformerBridge
from lora_transfer_pruning.adapter.torch_pruning.torch_pruning_group_builder import TorchPruningGroupBuilder
import torch_pruning as tp

from transformer_lens.model_bridge.generalized_components.linear import LinearBridge
from lora_transfer_pruning.usecase.torch_pruning_group_mapper import TorchPruningGroupMapper
from lora_transfer_pruning.core.prune_task_type import ModelPruneTask

class LocalPruning:
    #TODO not local? diff?
    def __init__(self, 
                 model_bridge: TransformerBridge, 
                 example_input_ids: torch.Tensor
        ):
        self.model_bridge = model_bridge
        unwrapped_params = []
        self.torch_pruning_group_builder = TorchPruningGroupBuilder(
            model_bridge, 
            example_input_ids, 
            unwrapped_params
        )

    def get_torch_pruning_groups_and_structural_setups(self, prune_task: ModelPruneTask) -> tuple[list[tp.Group], list[Callable[[], None]]]:
        '''Creates torch_pruning groups from prune_task without real pruning and fix indices in them (especially in attn).
        Also returns structural setups for each group, which are used for torch pruning to change inner model's constants.'''
        pruning_groups = []
        for module_name, groupPruneTask in prune_task.items():
            module = self.model_bridge.get_submodule(module_name)
            assert isinstance(module, LinearBridge), f"Module {module_name} is not a LinearBridge and cannot be pruned."
            if (groupPruneTask.cols is not None):
                pruning_groups.append(
                    self.torch_pruning_group_builder.get_correct_pruning_group_and_structural_setup(
                        module,
                        tp.prune_linear_in_channels,
                        groupPruneTask
                    )
                )
            if (groupPruneTask.rows is not None):
                pruning_groups.append(  
                    self.torch_pruning_group_builder.get_correct_pruning_group_and_structural_setup(
                        module,
                        tp.prune_linear_out_channels,
                        groupPruneTask
                    )
                )
        groups = [group for group, _ in pruning_groups]
        setups = [setup for _, setup in pruning_groups]
        return groups, setups
    
    def prepare_model_to_transfer_pruning(self, prune_task: ModelPruneTask, rescale=True):
        '''Prepares model for transfer pruning by creating torch_pruning groups,
        fixing indices in them and using these indices and groups for creating activation hooks on belonged modules.
        That hooks will zero out activations, implementing so called "transfer pruning".
        '''
        pruning_groups = self.get_torch_pruning_groups_and_structural_setups(prune_task)[0]
        self.prepare_model_to_transfer_pruning_from_groups(pruning_groups, rescale)
            
    def prepare_model_to_transfer_pruning_from_groups(self, pruning_groups: list[tp.Group], rescale=True):
        '''Prepares model for transfer from fixed torch_pruning groups'''
        for group in pruning_groups:
            TorchPruningGroupMapper.prepare_group_for_transfer_pruning(self.model_bridge, group, rescale)
        
        TorchPruningGroupMapper.prepare_all_norms(self.model_bridge)            