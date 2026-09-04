from typing import Any, cast

from peft import PeftModel
import torch
from transformers import TrainerCallback
from trl.trainer.sft_trainer import SFTTrainer

class _FreezeModelOnTrainBeginCallback(TrainerCallback):
    '''
    For multi adapter mode we can't update one ordinary weight multiple ways at the same time.
    Only adapters training allowed. One adapter - one prune task.
    '''
    
    def on_train_begin(self, args, state, control, **kwargs):
        kwargs["model"].requires_grad_(False)

class MultiAdapterPruningTrainer(SFTTrainer):
    '''
    Class for training many lora adapters in one model at the same time. Trains only adapters,
    other weights will be frozen.
    
    Important: train will touch adapters in ALL layers, despite their initial frozing.
    Please, provide adapters only on needed layers.
      
    If you want to train base weights without lora adapters but with transfer pruning,
    use default SFTTrainer with PeftModel or without Peft at all. 
    '''
    
    def __init__(
        self,
        model: PeftModel | None = None,
        *args,
        task_names: list[str] | None = None,
        **kwargs,
    ):
        if (not isinstance(model, PeftModel)):
            raise TypeError(f"Expected model to be an instance of PeftModel, but got {type(model)}")
        if (kwargs.get("peft_config") is not None):
            raise ValueError("peft_config should not be provided when using MultiAdapterPruningTrainer. Model should be already PEFT.")
        
        super().__init__(model=model, *args, **kwargs)
        self.add_callback(_FreezeModelOnTrainBeginCallback())
        if task_names is None:
            task_names = list(cast(PeftModel, self.model).peft_config.keys())
        self.task_names = task_names
        #for many adapters grad_norms summed, so we need to scale max_grad_norm
        self.args.max_grad_norm *= len(self.task_names) ** 0.5

    def training_step(
        self,
        model: PeftModel,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: torch.Tensor | int | None = None
    ):
        task_losses = {}

        for task_name in self.task_names:
            model.set_adapter(task_name)

            loss = super().training_step(
                model,
                inputs,
                num_items_in_batch=num_items_in_batch,
            )

            task_losses[task_name] = loss

        # Distinct values in log_history.
        # loss already normalized by Trainer on gradient accumulation.
        for task_name, loss in task_losses.items():
            self._metrics["train"][f"adapter_loss_{task_name}"].append(
                loss.detach().float().item()
            )

        # This used for general train_loss.
        # Backward was already called in super().training_step
        return torch.stack(tuple(task_losses.values())).mean()
    
        #optimizer.step() in Trainer._run_epoch() will update model params
        #if we froze all other params except adapter's, step will be correct
        
        #otherwise optimizer will update params with sum of .backward() of different adapters, which is not what we want
