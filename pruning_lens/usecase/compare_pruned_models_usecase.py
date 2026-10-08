from typing import Callable, Sequence

import torch
from transformer_lens.model_bridge.transformer_bridge import TransformerBridge

from lora_transfer_pruning.core.prune_task_type import ModelPruneTask
from lora_transfer_pruning.usecase.local_pruning import LocalPruning
from pruning_lens.adapter.torch_pruning.torch_pruning_tracer import TorchPruningTracer
from pruning_lens.adapter.transfer_pruning.transfer_pruning_tracer import TransferPruningTracer
from pruning_lens.core.compairing_trace import CompairingTrace
from pruning_lens.core.great_comparator import GreatComparator
from .hook_names_filter import HookNamesFilter
from pruning_lens.core.pruning_trace import PruningTrace

class ComparePrunedModelsUsecase:

    @staticmethod
    def compare_transfer_and_torch_prunings_on_layers(
        model: TransformerBridge,
        tokens: torch.Tensor,
        prune_task: ModelPruneTask,
        layers: list[int],
        compute_gradient: bool = False,
        *,
        atol: float = 1e-4,
        rtol: float = 1e-3,
        store_difference: bool = True,
        print_summary: bool = True,
        each_module_independently: bool = False,
    ) -> tuple[CompairingTrace, PruningTrace, PruningTrace]:
        '''Compare transfer pruning and torch_pruning traces on a subset of layers.
        
        Select hooks by layer prefix; intermediate hooks can be UNRESOLVED if we can't handle their defects (like shape change with torch pruning in hook_rot_q or hook_q_input).

        Unsupported TransformerLens conversions (inside HookPoints) can raise in the capture adapters.
        '''
        if not layers or any(layer < 0 for layer in layers):
            raise ValueError("layers must be a nonempty list of nonnegative layer indices")
                
        return ComparePrunedModelsUsecase.compare_transfer_and_torch_prunings(
            model, tokens, prune_task,
            names_filter=lambda name: name.startswith(tuple(f"blocks.{layer}." for layer in layers)),
            compute_gradient=compute_gradient,
            atol=atol, rtol=rtol,
            store_difference=store_difference, 
            print_summary=print_summary,
            each_module_independently=each_module_independently
        )

    @staticmethod
    def compare_transfer_and_torch_prunings(
        model: TransformerBridge,
        tokens: torch.Tensor,
        prune_task: ModelPruneTask,
        names_filter: str | Sequence[str] | Callable[[str], bool] | None = None,
        compute_gradient: bool = False,
        *,
        atol: float = 1e-4,
        rtol: float = 1e-3,
        store_difference: bool = True,
        print_summary: bool = True,
        each_module_independently: bool = False,
    ) -> tuple[CompairingTrace, PruningTrace, PruningTrace]:
        '''Capture SP then TP using ONE plan, including identical fractional indices.
        
        Support only <b>linear</b> hooks (hooks above linear layers), because only for them we can reconstruct shape defects (we see them as idxs inside torch_pruning groups).

        For names_filter selects only available hooks (not all from model.hook_dict) and only linear to correctly compare SP and TP.
        
        Mutates an initially unpruned model: it remains structurally pruned.
        Captures run in eval mode; original per-module training flags are restored.
        Non-permanent hooks are reset before comparison and after SP capture.
        TP setups remain installed. Structural changes are not rolled back on error.
        No copy of model weights is made; comparison differences are stored on CPU.

        names_filter=None selects affected linear hooks, not an upstream anchor.
        For gradients include a real grad-enabled upstream point on each branch
        (see BaseTracer). Missing gradients are reported as NO_GRAD, never OK.
        '''

        if torch.is_inference_mode_enabled():
            raise ValueError("Dependency graph construction needs autograd; leave inference_mode first")
        training_flags = [(module, module.training) for module in model.modules()]
        try:
            model.eval()
            model.reset_hooks()
            with torch.enable_grad():
                local_pruning = LocalPruning(model, tokens)
                prune_task_plan = local_pruning.get_torch_pruning_groups_and_structural_setups(prune_task)
                names_filter = HookNamesFilter.filter(model, 
                                                      prune_task_plan.groups,
                                                      names_filter,
                                                      only_with_grad = compute_gradient
                                                      )

                devices = sorted({
                    tensor.device.index
                    for tensor in [*model.parameters(), *model.buffers(), tokens]
                    if tensor.device.type == "cuda"
                })
                # Keep capture randomness identical and do not consume caller RNG.
                # Fractional index selection above still consumes its usual RNG.
                with torch.random.fork_rng(devices=devices):
                    cpu_state = torch.get_rng_state()
                    cuda_states = {device: torch.cuda.get_rng_state(device) for device in devices}
                    try:
                        local_pruning.prepare_model_to_transfer_pruning_from_groups(prune_task_plan.groups, rescale=False)
                        transfer_pruning_trace = TransferPruningTracer.capture_hookpoints_activations_with_defects(
                            model, tokens, names_filter=names_filter,
                            compute_gradient=compute_gradient, groups=prune_task_plan.groups,
                        ).to_cpu()
                    finally:
                        model.reset_hooks()

                    for setup in prune_task_plan.structural_setups:
                        setup()
                    for group in prune_task_plan.groups:
                        group.prune()

                    torch.set_rng_state(cpu_state)
                    for device, state in cuda_states.items():
                        torch.cuda.set_rng_state(state, device)
                    torch_pruning_trace = TorchPruningTracer.capture_hookpoints_activations_with_defects(
                        model, tokens, names_filter=names_filter,
                        compute_gradient=compute_gradient, groups=prune_task_plan.groups,
                    ).to_cpu()


            result = GreatComparator.compare_traces(
                torch_pruning_trace, transfer_pruning_trace, atol=atol, rtol=rtol,
                compute_gradient=compute_gradient,
                store_difference=store_difference,
            )
            if print_summary:
                result.print_summary()
            return result, transfer_pruning_trace, torch_pruning_trace
        finally:
            for module, training in training_flags:
                module.training = training
