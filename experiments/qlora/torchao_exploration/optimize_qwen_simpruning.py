"""Small opt-in changes for this Qwen SP experiment; no custom kernels.

Native attention deliberately bypasses TL's internal attention hooks, but keeps
the LinearBridge and RMSNorm children (where this experiment's SP hooks live).
Use only AFTER dependency graph construction, not while collecting attention traces.
"""
import torch
from transformer_lens.model_bridge.generalized_components.base import GeneralizedComponent
from transformer_lens.model_bridge.generalized_components.position_embeddings_attention import PositionEmbeddingsAttentionBridge
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge
from lora_transfer_pruning.core.tl_adapters import ExpertBridge


def prepare_async_rmsnorm(module):
    count = None

    def capture(x, hook):
        nonlocal count
        # Same first-vector convention as the existing hook, without CPU sync
        # or scanning the entire batch just to discard all but its first count.
        count = (x.reshape(-1, x.shape[-1])[0] == 0).sum()
        return x

    def correct(x, hook):
        coefficient = ((x.shape[-1] - count.double()) / x.shape[-1]).sqrt().float()
        # A GPU FP16 scalar would round the coefficient prematurely. Match
        # the original Python-scalar multiply's FP32 opmath, then cast once.
        return (x.float() * coefficient).to(x.dtype)

    module.hook_in.add_hook(capture)
    module.hook_out.add_hook(correct)


def replace_rms_hooks(model, factory):
    """Experiment owns these RMS hooks; do not use on a model with other norm hooks."""
    count = 0
    for module in model.modules():
        if isinstance(module, GeneralizedComponent) and 'rms' in type(module._original_component).__name__.lower():
            module.hook_in.remove_hooks()
            module.hook_out.remove_hooks()
            factory(module)
            count += 1
    return count


def use_native_sdpa(model):
    count = 0
    for module in model.modules():
        if isinstance(module, PositionEmbeddingsAttentionBridge):
            # Refuse to silently disable any live attention-level intervention.
            for child in module.children():
                if hasattr(child, 'fwd_hooks'):
                    assert not child.fwd_hooks and not child.bwd_hooks, child.name
            module._original_component.config._attn_implementation = 'sdpa'
            # GeneralizedComponent.__setattr__ forwards ordinary assignments
            # to HF; bind on the wrapper itself, not on its original component.
            object.__setattr__(module, 'forward', module._original_component.forward)
            count += 1
    return count


def bypass_inactive_bridges(model):
    """Training-only: skip empty Linear/Expert hook boundaries; retain SP hooks.

    Returns old forwards so callers can restore full tracing afterwards.
    """
    restore = []
    for module in model.modules():
        if isinstance(module, (LinearBridge, ExpertBridge)):
            points = [child for child in module.children() if hasattr(child, 'fwd_hooks')]
            if any(p.fwd_hooks or p.bwd_hooks for p in points):
                continue
            if any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks
                   for m in (module, module._original_component)):
                continue
            restore.append((module, module.forward))
            object.__setattr__(module, 'forward', module._original_component.forward)
    return restore


def skip_redundant_train(model):
    """Opt-in for this trainer: children change mode only via the root model.

    Still run the original recursive train/eval on every actual mode transition.
    Return the original method for restoring normal behavior after training.
    """
    original_train = model.train
    original_train(True)  # Synchronize all children before enabling the shortcut.

    def train(mode=True):
        if model.training != mode:
            original_train(mode)
        return model

    object.__setattr__(model, 'train', train)
    return original_train


def cache_flop_count(trainer, batch):
    """Cache the parameter count in the installed Trainer's FLOP estimate.

    Apply after wrapping/pruning. Restore the returned method before changing
    model structure. This preserves the existing estimate, not actual INT4 FLOPs.
    """
    original = trainer.floating_point_ops
    count = trainer.model.num_parameters(exclude_embeddings=True)
    key = getattr(trainer.model, 'main_input_name', 'input_ids')
    def estimate(inputs):
        return 6 * inputs[key].numel() * count if key in inputs else 0
    assert estimate(batch) == original(batch), 'Trainer FLOP formula changed'
    trainer.floating_point_ops = estimate
    return original


def cache_parameter_iteration(model):
    """Reuse the same Parameter references/order for clipping and zero_grad.

    No weights/gradients are copied. Restore the returned method before adding,
    removing or replacing parameters (adapters, structural pruning, device moves).
    Updating parameter values or requires_grad does not invalidate this tuple.
    """
    original = model.parameters
    parameters = tuple(original())
    object.__setattr__(model, 'parameters', lambda recurse=True:
                       iter(parameters) if recurse else original(recurse=False))
    return original
