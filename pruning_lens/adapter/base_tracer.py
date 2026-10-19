from abc import ABC, abstractmethod
from typing import Callable, Sequence

import torch
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
from .hook_names_filter import HookNamesFilter

class BaseTracer(ABC):
    
    @classmethod
    def capture_hookpoints_activations_with_defects(
        cls,
        model: TransformerBridge,
        tokens: torch.Tensor,
        names_filter: str | Sequence[str] | Callable[[str], bool] | None = None,
        compute_gradient: bool = True,
        groups: list[tp.Group] | None = None,
        local_gradients: bool = False
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
            compute_gradient: Whether to compute gradients for the captured hooks.
            local_gradients: Whether to compute local gradients (each loss for each module/hook point) for the captured hooks. 
                If True, it will use `cache_local_activation_and_gradient` instead of `cache_activation_and_gradient`.
                
                Note: grads for hook_out always be just ones. For compairing see hook_in.
        """
        
        if local_gradients:
            names_filter = HookNamesFilter.convert_to_only_models_names(model, names_filter)
            gradient_cache = BaseTracer._cache_local_activation_and_gradient(
                model,
                tokens,
                names_filter=names_filter,
            )
        else:
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
        '''Read TransformerLens layout defects for captured hooks (like shape changing in class `Attention` convert from [batch, seq, d_model] to [batch, seq, n_heads, d_head]) without changing cached tensors.
        
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
    def _cache_local_activation_and_gradient(
        model: TransformerBridge,
        tokens: torch.Tensor,
        names_filter: Sequence[str],
    ) -> GradientCache:
        """
        Capture local d(hook_out.sum()) / d(hook_in) using TL forward hooks.

        names_filter contains registered HOOK names, e.g.
        blocks.0.attn.q_proj.hook_in and blocks.0.attn.q_proj.hook_out.
        Both boundaries must be selected for each module. Aliases are resolved
        through hook_dict; cache keys are canonical hook names.
        Returned tensors are detached CPU snapshots in TL hook layouts.
        Repeated module invocations are rejected: GradientCache has only one
        entry per hook name and cannot represent separate local derivatives.

        gradients[hook_in] is d(local output.sum()) / d(input);
        gradients[hook_out] is ones_like(output), or None for a frozen output.
        metric is the FP64 sum of detached local output sums, for reporting only.
        Unlike end-to-end GradientCache, gradients are NOT derivatives of that
        aggregate metric: each pair has its own local objective.

        Capture runs after existing forward hooks (including SP masks). Thus
        grad_input is with respect to the post-hook input, NOT before an input
        mask. Inputs still come from the full model forward: differences in
        upstream activations are not eliminated by local differentiation.

        Capture returns a clone to the live path so converted tensors participate
        in autograd and each invocation has its own boundary. Requires grad-enabled
        inputs; frozen inputs or disconnected outputs yield grad_input=None.
        Parameter .grad buffers are not populated. Existing hooks are preserved.
        """
        if torch.is_inference_mode_enabled():
            raise ValueError("Local gradient capture cannot run inside inference_mode")
        if isinstance(names_filter, str) or not names_filter:
            raise ValueError("names_filter must be a nonempty sequence of hook names")

        calls = []
        pending = {}
        fwd_hooks = []
        seen_hooks = set()
        boundaries = {}

        def make_hook(name, direction):
            def capture(tensor, hook):
                live = tensor.clone()
                snapshot = tensor.detach().cpu().clone()
                if direction == "hook_in":
                    call = {"name": name, "x": live, "input": snapshot}
                    calls.append(call)
                    pending[name].append(call)
                else:
                    if not pending[name]:
                        raise RuntimeError(f"{name}: hook_out fired without hook_in")
                    call = pending[name].pop()
                    call.update(y=live, output=snapshot)
                # None would leave a converted side view outside the loss path.
                return live
            return capture

        for hook_name in names_filter:
            point = model.hook_dict.get(hook_name)
            if point is None or point.name is None:
                raise ValueError(f"Unknown registered hook: {hook_name}")
            name, _, direction = point.name.rpartition(".")
            if direction not in ("hook_in", "hook_out"):
                raise ValueError(f"{hook_name}: local capture requires hook_in/hook_out boundaries")
            if id(point) in seen_hooks:
                continue
            seen_hooks.add(id(point))
            boundaries.setdefault(name, set()).add(direction)
            fwd_hooks.append((point.name, make_hook(name, direction)))

        for name, directions in boundaries.items():
            missing = {"hook_in", "hook_out"} - directions
            if missing:
                raise ValueError(f"{name}: names_filter is missing {', '.join(sorted(missing))}")
            pending[name] = []

        try:
            with torch.enable_grad(), model.hooks(fwd_hooks=fwd_hooks):
                model(tokens)
                for name in boundaries:
                    if pending[name] or not any(call["name"] == name for call in calls):
                        raise RuntimeError(f"{name}: hook_in/hook_out did not both execute")
                    if sum(call["name"] == name for call in calls) != 1:
                        raise ValueError(f"{name}: repeated calls cannot be stored in GradientCache")

                activations = {}
                gradients = {}
                metric = torch.zeros((), dtype=torch.float64)
                for call in calls:
                    x, y = call["x"], call["y"]
                    grad_input = None

                    if x.requires_grad and y.requires_grad:
                        grad_input, = torch.autograd.grad(
                            outputs=y,
                            inputs=x,
                            grad_outputs=torch.ones_like(y),  # как y.sum()
                            retain_graph=True,
                            allow_unused=True,
                        )

                    prefix = call["name"] + "." if call["name"] else ""
                    in_name, out_name = prefix + "hook_in", prefix + "hook_out"
                    activations[in_name] = call["input"]
                    activations[out_name] = call["output"]
                    gradients[in_name] = (
                        None if grad_input is None
                        else grad_input.detach().cpu().clone()
                    )
                    gradients[out_name] = (
                        torch.ones_like(call["output"]) if y.requires_grad else None
                    )
                    metric += call["output"].double().sum()

                return GradientCache(activations=activations, gradients=gradients, metric=metric)
        finally:
            pending.clear()
            calls.clear()  # Release references to the live graph.
