from typing import Callable, Sequence

import torch
from transformer_lens.model_bridge.transformer_bridge import TransformerBridge
from lora_transfer_pruning.core.prune_task_type import ModelPruneTask
from lora_transfer_pruning.usecase.local_pruning import LocalPruning
from pruning_lens.adapter.transfer_pruning.transfer_pruning_tracer import TransferPruningTracer
from pruning_lens.adapter.torch_pruning.torch_pruning_tracer import TorchPruningTracer
from pruning_lens.core.pruning_trace import PruningTrace
from pruning_lens.core.hook_names_filter import HookNamesFilter

class PruneAndCaptureUsecase:
    
    #how to use names_filter:
    # For compute_gradient=True, these selections also need a real grad-enabled
    # upstream hook for each traced branch (see method docstrings below).
    # For one attention block, a possible anchor name is:
    # upstream_name = model.blocks[layer].attn.hook_in.name
    # names_filter = [upstream_name, model.blocks[layer].attn.q.hook_out.name]
    # For a predicate, include the anchor explicitly with `name == upstream_name or ...`.
    #     # 1. Несколько конкретных точек
    # names_filter = [
    #     "blocks.0.attn.hook_z",
    #     "blocks.0.hook_mlp_out",
    #     "blocks.5.hook_mlp_out",
    # ]

    # # 2. Выходы MLP во всех слоях
    # names_filter = lambda name: name.endswith(".hook_mlp_out")

    # # 3. Выходы attention heads во всех слоях
    # names_filter = lambda name: name.endswith(".attn.hook_z")

    # # 4. Выходы MLP только в выбранных слоях
    # layers = {0, 3, 7}
    # names_filter = [
    #     f"blocks.{layer}.hook_mlp_out"
    #     for layer in layers
    # ]

    # # 5. Несколько типов точек
    # names_filter = lambda name: name.endswith(
    #     (".attn.hook_z", ".hook_mlp_out")
    # )
    
    
    @staticmethod
    def get_transfer_pruning_trace(model: TransformerBridge,
                                tokens: torch.Tensor,
                                prune_task: ModelPruneTask,
                                compute_gradient: bool = True,
                                names_filter: str | Sequence[str] | Callable[[str], bool] | None = None
                                ) -> PruningTrace:
        '''Prepare simulation pruning in place and capture activations/gradients.

        None in names_filter selects affected linear boundaries from the resolved prune_task groups,
        including dependencies, not just the task's root modules. Explicit filters
        are passed through unchanged. Uses the currently active adapter (lora), if any.
        Pass a fresh, unpruned model; existing hooks are not reset here.

        Warning:
            With compute_gradient=True, names_filter must include a real,
            grad-enabled upstream hook on the loss path for each traced branch.
            TL capture can return None gradients when only converted views are
            selected. Auto-selection (names_filter=None) does not add an anchor.
            For one attention block, its attn.hook_in.name is a candidate.
            See BaseTracer for the full capture limitation.
            Activation-only capture (compute_gradient=False) needs no anchor.
        '''
        localPruning = LocalPruning(model, tokens)
        prune_task_plan = localPruning.get_torch_pruning_groups_and_structural_setups(prune_task)
        names_filter = HookNamesFilter.filter(model, prune_task_plan.groups, names_filter)
        # This is a single-model trace, not a named multi-adapter training plan.
        localPruning.prepare_model_to_transfer_pruning_from_groups(
            prune_task_plan.groups, rescale=False
        )
        return TransferPruningTracer.capture_hookpoints_activations_with_defects(
            model,
            tokens,
            names_filter=names_filter,
            compute_gradient=compute_gradient,
            groups=prune_task_plan.groups,
        )

    @staticmethod
    def get_torch_pruning_trace(
        model: TransformerBridge,
        tokens: torch.Tensor,
        prune_task: ModelPruneTask,
        compute_gradient: bool = True,
        names_filter: str | Sequence[str] | Callable[[str], bool] | None = None,
    ) -> PruningTrace:
        '''Apply structural pruning in place and capture its trace.

        None for names_filter selects the same affected linear hooks as the simulation method.
        Use a separate fresh model when comparing TP and simulation pruning.

        Warning:
            With compute_gradient=True, names_filter must include a real,
            grad-enabled upstream hook on the loss path for each traced branch.
            TP does not guarantee that TL conversions are skipped. Auto-selection
            (names_filter=None) does not add an anchor, so converted-only capture
            can return None gradients. A block's attn.hook_in.name is a candidate
            for example.
            Activation-only capture (compute_gradient=False) needs no anchor.
        '''
        localPruning = LocalPruning(model, tokens)
        prune_task_plan = localPruning.get_torch_pruning_groups_and_structural_setups(prune_task)
        names_filter = HookNamesFilter.filter(model, prune_task_plan.groups, names_filter)

        for setup in prune_task_plan.structural_setups:
            setup()
        for group in prune_task_plan.groups:
            group.prune()

        return TorchPruningTracer.capture_hookpoints_activations_with_defects(
            model,
            tokens,
            names_filter=names_filter,
            compute_gradient=compute_gradient,
            groups=prune_task_plan.groups,
        )


