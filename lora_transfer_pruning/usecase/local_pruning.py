from peft import PeftModel
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge
from lora_transfer_pruning.adapter.torch_pruning.torch_pruning_group_builder import TorchPruningGroupBuilder
import torch_pruning as tp

from transformer_lens.model_bridge.generalized_components.linear import LinearBridge
from lora_transfer_pruning.usecase.torch_pruning_group_mapper import TorchPruningGroupMapper
from lora_transfer_pruning.core.prune_task_type import ModelPruneTask
from .prune_task_plan import PruneTaskPlan


class LocalPruning:
    def __init__(self,
                 model_bridge: TransformerBridge,
                 example_input_ids: torch.Tensor,
                 peft_model: PeftModel | None = None,
                 ):
        '''Initialize LocalPruning with a model bridge, example input IDs (to build dependency graph), and an optional peft_model,
        if you plan to use many prune tasks and lora adapters for full transfer pruning.'''
        self.model_bridge = model_bridge
        unwrapped_params = []
        self.torch_pruning_group_builder = TorchPruningGroupBuilder(
            model_bridge,
            example_input_ids,
            unwrapped_params
        )
        self.peft_model = peft_model
        self.active_adapters_source = None
        if peft_model is not None:
            self.active_adapters_source = lambda: peft_model.active_adapters

    def get_torch_pruning_groups_and_structural_setups(self, prune_task: ModelPruneTask) -> PruneTaskPlan:
        '''Creates torch_pruning groups from prune_task without real pruning and fix indices in them (especially in attn).
        Also returns structural setups for each group, which are used for torch pruning to change inner model's constants.'''
        pruning_groups = []
        for module_name, groupPruneTask in prune_task.data.items():
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
        return PruneTaskPlan(prune_task.name, groups, setups)

    def get_full_transfer_pruning_plan(self, prune_tasks: list[ModelPruneTask]) -> list[PruneTaskPlan]:
        '''Creates torch_pruning groups from prune_task without real pruning and fix indices in them (especially in attn).
        Also returns structural setups for each group, which are used for torch pruning to change inner model's constants.'''
        plans = []
        for prune_task in prune_tasks:
            plans.append(self.get_torch_pruning_groups_and_structural_setups(prune_task))
        return plans
    
    def _activate_model_adapters_grads_for_future_optimizer(self, prune_task_plans: list[PruneTaskPlan]):
        '''Activates gradients for model's lora adapters, which are used in prune_task_plans.'''
        if (len(prune_task_plans) > 1):
            if (self.peft_model is None):
                raise ValueError("If you want to use multiple prune tasks and lora adapters for full transfer pruning, you must provide a peft_model.")
                        
            prepare_adapter_names = []
            for prune_task_plan in prune_task_plans:
                if (prune_task_plan.name is None):
                    raise ValueError("If you want to use multiple prune tasks and lora adapters for full transfer pruning, you must provide a name for each prune task.")
                prepare_adapter_names.append(prune_task_plan.name)
                
            self.peft_model.set_requires_grad(prepare_adapter_names, requires_grad=True)
        elif (len(prune_task_plans) == 1):
            if (self.peft_model is None):
                if (prune_task_plans[0].name is not None):
                    raise ValueError("Named prune task plan is provided, but peft_model is None. If you want to use a named prune task plan, you must provide a peft_model.")
            else:
                if (prune_task_plans[0].name is None):
                    raise ValueError("Unnamed prune task plan is provided, but peft_model is not None. If you want to use a peft_model, you must provide a name for the prune task plan.")
                self.peft_model.set_requires_grad([prune_task_plans[0].name], requires_grad=True)
    
    def prepare_model_to_transfer_pruning(self, prune_task_plans: list[PruneTaskPlan], rescale=True):
        '''Prepares model to transfer pruning by creating torch_pruning groups,
        fixing indices in them and using these indices and groups for creating activation hooks on belonged modules.
        That hooks will zero out activations, implementing so called "transfer pruning".
        '''
        
        self._activate_model_adapters_grads_for_future_optimizer(prune_task_plans)
        
        for prune_task_plan in prune_task_plans:
            self._prepare_model_to_transfer_pruning_from_groups(prune_task_plan.groups,
                                                                rescale,
                                                                prune_task_plan.name)
            
        TorchPruningGroupMapper.prepare_all_norms(self.model_bridge)

    def _prepare_model_to_transfer_pruning_from_groups(self, pruning_groups: list[tp.Group], rescale, prune_task_name: str | None):
        '''Prepares model for transfer from fixed torch_pruning groups'''
        for group in pruning_groups:
            TorchPruningGroupMapper.prepare_group_for_transfer_pruning(self.model_bridge,
                                                                       group,
                                                                       rescale=rescale,
                                                                       prune_task_name=prune_task_name,
                                                                       active_adapters_source=self.active_adapters_source)
    
    def prepare_model_to_transfer_pruning_from_groups(self, groups: list[tp.Group], rescale=True):
        '''Prepares model for transfer from fixed torch_pruning groups (linked to one prune task)
        and without bounding ablation hooks to model's active lora adapters (aka prune_task_name)'''
        self.prepare_model_to_transfer_pruning([PruneTaskPlan(None, groups, [])], rescale=rescale)
