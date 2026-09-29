from typing import cast

import torch

from transformer_lens.tools.analysis.attribution_patching import GradientCache
from transformer_lens.model_bridge.transformer_bridge import TransformerBridge

from pruning_lens.core.pruning_trace import PruningTrace
from pruning_lens.adapter.base_tracer import BaseTracer
import torch_pruning as tp
from pruning_lens.core.activation_corruption import ActivationCorruption


class TransferPruningTracer(BaseTracer):
    '''Trace simulation pruning using TL capture and inverse layout metadata.
    '''

    @staticmethod
    def _append_defects(
        gradient_cache: GradientCache, 
        model: TransformerBridge,
        groups: list[tp.Group],
        corruptions_for_all_hookpoints: dict[str, ActivationCorruption],
        compute_gradient: bool
    ) -> PruningTrace:
        """Attach inverse layout metadata without modifying cached tensor values."""

        if (compute_gradient):
            #correct gradients from grad incoming in (x * mask) to grad incoming in x (grad = grad * mask)
            for hook, idxs in BaseTracer.iter_pruned_linear_hooks(model, groups):
                if (hook.name is None or hook.name not in corruptions_for_all_hookpoints):
                    continue
                if (gradient_cache.gradients[hook.name] is not None):
                    cast(torch.Tensor, gradient_cache.gradients[hook.name])[idxs] = 0.0
        
        return PruningTrace(
            activations=gradient_cache.activations,
            gradients=gradient_cache.gradients,
            metric=gradient_cache.metric,
            corruptions=corruptions_for_all_hookpoints,
        )
