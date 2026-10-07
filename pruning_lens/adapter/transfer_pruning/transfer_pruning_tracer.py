import torch

from transformer_lens.tools.analysis.attribution_patching import GradientCache
from transformer_lens.model_bridge.transformer_bridge import TransformerBridge

from pruning_lens.core.pruning_trace import PruningTrace
from pruning_lens.adapter.base_tracer import BaseTracer
import torch_pruning as tp
from pruning_lens.core.activation_corruption import ActivationCorruption
from lora_transfer_pruning.adapter.torch_pruning.tp_utils import iter_pruned_linear_hooks


class TransferPruningTracer(BaseTracer):
    '''Trace simulation pruning using TL capture and inverse layout metadata.
    '''

    @staticmethod
    def _append_defects(
        gradient_cache: GradientCache, 
        model: TransformerBridge,
        groups: list[tp.Group],
        corruptions_upon_gradient_cache: dict[str, ActivationCorruption],
        compute_gradient: bool
    ) -> PruningTrace:
        """Attach layout metadata and mask gradients in native channel coordinates."""

        if (compute_gradient):
            #correct gradients from grad incoming in (x * mask) to grad incoming in x (grad = grad * mask)
            for hook, idxs in iter_pruned_linear_hooks(model, groups):
                if (hook.name is None or hook.name not in corruptions_upon_gradient_cache):
                    continue
                gradient = gradient_cache.gradients.get(hook.name)
                if gradient is None:
                    continue
                native_gradient = corruptions_upon_gradient_cache[hook.name].restore_activation(gradient).clone()
                indices = torch.as_tensor(idxs, device=native_gradient.device, dtype=torch.long)
                native_gradient.index_fill_(-1, indices, 0.0)
                # Keep the cached layout: the comparator restores both activations
                # and gradients using the same corruption metadata.
                gradient_cache.gradients[hook.name] = (
                    hook.hook_conversion.convert(native_gradient)
                    if hook.hook_conversion is not None else native_gradient
                )
        
        return PruningTrace(
            activations=gradient_cache.activations,
            gradients=gradient_cache.gradients,
            metric=gradient_cache.metric,
            corruptions=corruptions_upon_gradient_cache,
        )
