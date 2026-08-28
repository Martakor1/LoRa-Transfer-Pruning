import torch
from trl import SFTTrainer


class MultiAdapterPruningTrainer(SFTTrainer):
    def __init__(
        self,
        *args,
        pruning_controller: PruningTaskController,
        task_names: tuple[str, ...],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pruning_controller = pruning_controller
        self.task_names = task_names

    def training_step(
        self,
        model,
        inputs,
        num_items_in_batch=None,
    ):
        task_losses = {}

        for task_name in self.task_names:
            model.set_adapter(task_name)

            with self.pruning_controller.use(task_name):
                # shallow copy защищает от возможных изменений словаря
                loss = super().training_step(
                    model,
                    dict(inputs),
                    num_items_in_batch=num_items_in_batch,
                )

            task_losses[task_name] = loss

        # Отдельные значения в log_history.
        # loss уже нормализован Trainer по gradient accumulation.
        for task_name, loss in task_losses.items():
            self._metrics["train"][f"loss_{task_name}"].append(
                loss.detach().float().item()
            )

        # Это значение используется для общего train_loss.
        # Backward уже был выполнен отдельно для каждого адаптера.
        return torch.stack(tuple(task_losses.values())).mean()