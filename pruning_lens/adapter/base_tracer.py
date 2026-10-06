from abc import ABC, abstractmethod
from typing import Callable, Iterator, Sequence, cast

import torch
from torch import nn
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge
from transformer_lens.tools.analysis import attribution_patching
from transformer_lens.tools.analysis.attribution_patching import GradientCache
from transformer_lens.model_bridge.transformer_bridge import TransformerBridge

from pruning_lens.core.pruning_trace import PruningTrace
from pruning_lens.core.activation_corruption import ActivationCorruption
from pruning_lens.core.transformer_lens_conversion_registry import (
    TransformerLensConversionRegistry,
    UnsupportedHookConversionError,
)
import torch_pruning as tp
from lora_transfer_pruning.adapter.torch_pruning.tp_utils import get_active_module_in_dep_graph_from_linear_bridge

class BaseTracer(ABC):
    
    @classmethod
    def capture_hookpoints_activations_with_defects(
        cls,
        model: TransformerBridge,
        tokens: torch.Tensor,
        names_filter: str | Sequence[str] | Callable[[str], bool] | None = None,
        compute_gradient: bool = True,
        groups: list[tp.Group] | None = None,
    ) -> PruningTrace:
        """Capture an already SP-prepared or TP-prepared model and snapshot TL layout defects.

        Supports only capturing usual TransformerLens models and TL models with LoRA adapters considering only one active adapter!
        
        Warning:
            With compute_gradient=True, names_filter must also select a real,
            grad-enabled upstream tensor on the loss path for each traced branch.
            TL currently drives backward through captured tensors: selecting only
            converted side views can leave gradients as None. A final output hook
            does not force backward through earlier layers. This requirement does
            not apply to activation-only capture (compute_gradient=False).

            attn.hook_in can serve as an anchor for that attention block when its
            input is [B, S, hidden] and requires_grad=True: AttentionAutoConversion
            returns that 3D tensor unchanged. The conversion object need not be
            None; it must leave the captured tensor on the actual loss path.
            Use the canonical name from model.blocks[layer].attn.hook_in.
            Frozen inputs, 4D inputs and separate branches need explicit checking;
            neither the hook name nor names_filter=None guarantees this condition.
        
        Args:
            groups: Complete list of already applied TP groups for the same model.
                None means pruning history is unknown; [] explicitly means no pruning.
                TorchPruningTracer supports one pruning operation per linear boundary.
            names_filter: String, sequence of names, or predicate selecting hooks.
                None uses TL's default subset, not all hooks. See `cache_activation_and_gradient`.
        """

        def metric_fn(logits: torch.Tensor) -> torch.Tensor:
            return model.loss_fn(
                logits,
                tokens,
            )

        gradient_cache = attribution_patching.cache_activation_and_gradient(
            model,
            tokens,
            metric_fn,
            names_filter=names_filter,
            compute_gradient=compute_gradient,
        )
        
        corruptions = BaseTracer._get_conversion_corruptions(gradient_cache, model)
        
        if groups is None:
            return PruningTrace(
                activations=gradient_cache.activations,
                gradients=gradient_cache.gradients,
                metric=gradient_cache.metric,
                corruptions=corruptions,
                unresolved_pruning_hooks=set(),
            )
        else:
            return cls._append_defects(
                gradient_cache, 
                model, 
                groups=groups,
                corruptions_upon_gradient_cache=corruptions,
                compute_gradient=compute_gradient
            )

    @staticmethod
    def _get_conversion_corruptions(
        gradient_cache: GradientCache,
        model: TransformerBridge,
    ) -> dict[str, ActivationCorruption]:
        '''Read TransformerLens layout defects for captured hooks (like shape changing in Attention Convert from [batch, seq, d_model] to [batch, seq, n_heads, d_head]) without changing cached tensors.
        
        Returns corruptions for all hookpoints. If no corruption is present, the defects list is empty inside ActivationCorruption.
        '''
        corruptions = {}
        for name, activation in gradient_cache.activations.items():
            if name not in model.hook_dict:
                raise KeyError(f"Captured hook {name!r} is missing from model.hook_dict")
            conversion = model.hook_dict[name].hook_conversion
            try:
                defects = TransformerLensConversionRegistry.map_conversion_to_defects(
                    conversion, tuple(activation.shape)
                )
            except UnsupportedHookConversionError as exc:
                raise UnsupportedHookConversionError(f"Hook {name!r}: {exc}") from exc
            corruptions[name] = ActivationCorruption(defects=defects)
        return corruptions
    
    @staticmethod
    @abstractmethod
    def _append_defects(
        gradient_cache: GradientCache,
        model: TransformerBridge,
        groups: list[tp.Group],
        corruptions_upon_gradient_cache: dict[str, ActivationCorruption],
        compute_gradient: bool
    ) -> PruningTrace:
       pass

    @staticmethod
    def iter_pruned_linear_hooks(
        model: TransformerBridge,
        groups: list[tp.Group],
    ) -> Iterator[tuple[HookPoint, list[int]]]:
        '''Resolve group operations to canonical outer linear hooks and their indices.

        Shared by automatic capture selection and TP defect collection.
        LoRA uses the active A input / B output, not internal rank boundaries.
        '''
        visited_hooks = set()
        for group in groups:
            DG = cast(tp.DependencyGraph, group._DG)
            for dep, idxs in group:
                module = dep.target.module
                if not isinstance(module, nn.Linear) or len(idxs) == 0:
                    continue
                if DG.is_out_channel_pruning_fn(dep.handler):
                    pruning_fn = tp.prune_linear_out_channels
                elif DG.is_in_channel_pruning_fn(dep.handler):
                    pruning_fn = tp.prune_linear_in_channels
                else:
                    continue

                # DG stores the module path separately from its human-readable repr.
                module_name = DG._module2name.get(module, "")
                bridge_name, separator, _ = module_name.rpartition("._original_component")
                if not separator:
                    continue
                component = model.get_submodule(bridge_name)
                if not isinstance(component, LinearBridge):
                    continue
                try:
                    selected_module = get_active_module_in_dep_graph_from_linear_bridge(
                        component, pruning_fn
                    )
                except TypeError:
                    continue
                if module is not selected_module:
                    continue  # Skip base/other adapters and internal LoRA-rank operations.

                hook = component.hook_in if pruning_fn == tp.prune_linear_in_channels else component.hook_out
                if hook.name in visited_hooks:
                    raise ValueError(
                        "Repeated channel pruning for the same linear boundary; "
                        "sequential coordinate mappings are not supported"
                    )
                visited_hooks.add(hook.name)
                
                yield hook, idxs
