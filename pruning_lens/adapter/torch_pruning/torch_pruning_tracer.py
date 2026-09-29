from typing import Iterator, cast
import warnings

import torch
import torch.nn as nn
import torch_pruning as tp
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.transformer_bridge import TransformerBridge
from transformer_lens.tools.analysis.attribution_patching import GradientCache

from lora_transfer_pruning.adapter.torch_pruning.tp_utils import get_active_module_in_dep_graph_from_linear_bridge
from pruning_lens.adapter.base_tracer import BaseTracer
from pruning_lens.core.activation_corruption import ActivationCorruption, RemovedChannelsDefect
from pruning_lens.core.pruning_trace import PruningTrace


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
        corruptions_for_all_hookpoints: dict[str, ActivationCorruption],
        compute_gradient: bool
    ) -> PruningTrace:
        '''Combine direct linear pruning defects with TL conversion defects.
        '''
        unresolved_pruning_hooks = set(gradient_cache.activations)

        if groups is not None:
            unresolved_pruning_hooks.difference_update( #TODO drop check for unresolved hooks or ban them in gradient_cache or what?
                TorchPruningTracer._get_linear_hook_names(model)
            )
            TorchPruningTracer._append_pruning_group_defects(model, groups, corruptions_for_all_hookpoints)

        if unresolved_pruning_hooks:
            reason = (
                "TP groups were not provided"
                if groups is None
                else "not a canonical supported linear input/output hook"
            )
            warnings.warn(
                f"Pruning mapping is unknown ({reason}) for "
                f"{len(unresolved_pruning_hooks)} captured hooks: "
                + ", ".join(sorted(unresolved_pruning_hooks))
                + ". Their tensors and known TL conversion defects are retained.",
                UnknownPruningMappingWarning,
                stacklevel=2,
            )

        return PruningTrace(
            activations=gradient_cache.activations,
            gradients=gradient_cache.gradients,
            metric=gradient_cache.metric,
            corruptions=corruptions_for_all_hookpoints,
            unresolved_pruning_hooks=unresolved_pruning_hooks,
        )

    @staticmethod
    def _get_linear_hook_names(
        model: TransformerBridge,
    ) -> set[str]:
        '''Recognize linear boundaries independently of whether they were pruned.'''
        hook_names = set()
        for component in model.modules():
            if not isinstance(component, LinearBridge) or component.original_component is None:
                continue
            for hook, pruning_fn in (
                (component.hook_in, tp.prune_linear_in_channels),
                (component.hook_out, tp.prune_linear_out_channels),
            ):
                try:
                    get_active_module_in_dep_graph_from_linear_bridge(component, pruning_fn)
                except TypeError:
                    # Unsupported LinearBridge backends remain unresolved.
                    continue
                if hook.name is not None:
                    hook_names.add(hook.name)
        return hook_names

    @staticmethod
    def _append_pruning_group_defects(
        model: TransformerBridge,
        groups: list[tp.Group],
        corruptions_for_all_hookpoints: dict[str, ActivationCorruption],
    ) -> None:
        '''Visit dependencies and attach their indices directly to matching captured hooks.

        Sequential pruning needs composed coordinates, so repeated operations
        are rejected instead of silently merging indices.
        '''
        
        for hook, idxs in BaseTracer.iter_pruned_linear_hooks(model, groups):
            if hook.name not in corruptions_for_all_hookpoints:
                continue
            removed_indices = torch.tensor(
                sorted(set(map(int, idxs))), dtype=torch.long
            )
            # First undo TL conversions, then restore removed channels.
            corruptions_for_all_hookpoints[hook.name].defects.insert(0, RemovedChannelsDefect(removed_indices))

    
