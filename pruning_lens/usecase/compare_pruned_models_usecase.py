from typing import Callable, Sequence

import torch
from transformer_lens.model_bridge.transformer_bridge import TransformerBridge

from lora_transfer_pruning.core.prune_task_type import ModelPruneTask
from lora_transfer_pruning.usecase.local_pruning import LocalPruning
from pruning_lens.adapter.torch_pruning.torch_pruning_tracer import TorchPruningTracer
from pruning_lens.adapter.transfer_pruning.transfer_pruning_tracer import TransferPruningTracer
from pruning_lens.core.compairing_trace import CompairingTrace
from pruning_lens.core.great_comparator import GreatComparator
from pruning_lens.core.pruning_trace import PruningTrace
from .prune_and_capture_usecase import PruneAndCaptureUsecase


class ComparePrunedModelsUsecase:

    @staticmethod
    def compare_transfer_and_torch_prunings_on_layers(
        model: TransformerBridge,
        tokens: torch.Tensor,
        prune_task: ModelPruneTask,
        layers: list[int],
        compute_gradient: bool = True,
        *,
        atol: float = 1e-4,
        rtol: float = 1e-3,
        store_difference: bool = True,
        print_summary: bool = True,
    ) -> CompairingTrace:
        '''Select hooks by layer prefix; intermediate pruning mappings can be UNRESOLVED.

        This is not an architecture-specific alignment of every hook in a layer.
        Unsupported TL conversions still raise in the capture adapters.
        '''
        if not layers or any(layer < 0 for layer in layers):
            raise ValueError("layers must be a nonempty list of nonnegative layer indices")
        prefixes = tuple(f"blocks.{layer}." for layer in layers)
        return ComparePrunedModelsUsecase.compare_transfer_and_torch_prunings(
            model, tokens, prune_task,
            name_filter=lambda name: name.startswith(prefixes),
            compute_gradient=compute_gradient,
            atol=atol, rtol=rtol,
            store_difference=store_difference, print_summary=print_summary,
        )

    @staticmethod
    def compare_transfer_and_torch_prunings(
        model: TransformerBridge,
        tokens: torch.Tensor,
        prune_task: ModelPruneTask,
        name_filter: str | Sequence[str] | Callable[[str], bool] | None = None,
        compute_gradient: bool = True,
        *,
        atol: float = 1e-4,
        rtol: float = 1e-3,
        store_difference: bool = True,
        print_summary: bool = True,
    ) -> CompairingTrace:
        '''Capture SP then TP using ONE plan, including identical fractional indices.

        Mutates an initially unpruned model: it remains structurally pruned.
        Captures run in eval mode; original per-module training flags are restored.
        Non-permanent hooks are reset before comparison and after SP capture.
        TP setups remain installed. Structural changes are not rolled back on error.
        No copy of model weights is made; comparison differences are stored on CPU.

        name_filter=None selects affected linear hooks, not an upstream anchor.
        For gradients include a real grad-enabled upstream point on each branch
        (see BaseTracer). Missing gradients are reported as NO_GRAD, never OK.
        Repeated pruning of the same parameter boundary is not supported.
        '''

        if torch.is_inference_mode_enabled():
            raise ValueError("Dependency graph construction needs autograd; leave inference_mode first")
        training_flags = [(module, module.training) for module in model.modules()]
        try:
            model.eval()
            model.reset_hooks()
            with torch.enable_grad():
                local_pruning = LocalPruning(model, tokens)
                plan = local_pruning.get_torch_pruning_groups_and_structural_setups(prune_task)
                names = PruneAndCaptureUsecase._get_trace_names_filter(model, plan.groups, name_filter)

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
                        local_pruning.prepare_model_to_transfer_pruning_from_groups(plan.groups, rescale=False)
                        candidate = TransferPruningTracer.capture_hookpoints_activations_with_defects(
                            model, tokens, names_filter=names,
                            compute_gradient=compute_gradient, groups=plan.groups,
                        ).to_cpu()
                    finally:
                        model.reset_hooks()

                    for setup in plan.structural_setups:
                        setup()
                    for group in plan.groups:
                        group.prune()

                    torch.set_rng_state(cpu_state)
                    for device, state in cuda_states.items():
                        torch.cuda.set_rng_state(state, device)
                    reference = TorchPruningTracer.capture_hookpoints_activations_with_defects(
                        model, tokens, names_filter=names,
                        compute_gradient=compute_gradient, groups=plan.groups,
                    ).to_cpu()


            result = GreatComparator.compare_traces(
                reference, candidate, atol=atol, rtol=rtol,
                compute_gradient=compute_gradient,
                store_difference=store_difference,
            )
            if print_summary:
                result.print_summary()
            return result
        finally:
            for module, training in training_flags:
                module.training = training


        
        
