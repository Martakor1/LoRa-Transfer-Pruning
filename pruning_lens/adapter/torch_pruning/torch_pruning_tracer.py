import torch
import torch_pruning as tp
from transformer_lens.model_bridge.transformer_bridge import TransformerBridge
from transformer_lens.tools.analysis.attribution_patching import GradientCache

from pruning_lens.adapter.base_tracer import BaseTracer
from pruning_lens.core.activation_corruption import ActivationCorruption, RemovedChannelsDefect
from pruning_lens.core.pruning_trace import PruningTrace
from lora_transfer_pruning.adapter.torch_pruning.tp_utils import iter_pruned_linear_hooks


class UnknownPruningMappingWarning(UserWarning):
    '''Captured hooks whose removed coordinates cannot be inferred from TP groups.'''


class TorchPruningTracer(BaseTracer):
    '''Trace structural pruning with removed-channel and TL layout metadata.
    '''

    @staticmethod
    def _append_defects(
        gradient_cache: GradientCache,
        model: TransformerBridge,
        groups: list[tp.Group],
        corruptions_upon_gradient_cache: dict[str, ActivationCorruption],
        compute_gradient: bool
    ) -> PruningTrace:
        '''Combine direct linear pruning defects with TL conversion defects.
        '''

        if groups is not None:
            for hook, idxs in iter_pruned_linear_hooks(model, groups):
                if hook.name not in corruptions_upon_gradient_cache:
                    continue
                
                removed_indices = torch.tensor(
                    sorted(set(map(int, idxs))), dtype=torch.long
                )
                # First undo TL conversions, then restore removed channels.
                corruptions_upon_gradient_cache[hook.name].defects.insert(0, RemovedChannelsDefect(removed_indices))

        return PruningTrace(
            activations=gradient_cache.activations,
            gradients=gradient_cache.gradients,
            metric=gradient_cache.metric,
            corruptions=corruptions_upon_gradient_cache
        )
