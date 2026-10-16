"""Explicit TL hook-conversion -> layout-defect mapping (without observers).

Projection head conversions assume native [B, S, H*D] inputs; OPT token
unflattening assumes native [B*S, D]. Already-expanded native inputs cannot
be distinguished from converted inputs after capture. These rules describe
those bridge contracts, not an arbitrary conversion's execution history.
Only layout bijections are supported, so the same inverse applies to gradients.
"""

import torch
from transformer_lens.conversion_utils.conversion_steps.base_tensor_conversion import (
    BaseTensorConversion,
)

from pruning_lens.core.activation_corruption import (
    Defect,
    RearrangeDefect,
    ReshapeDefect,
    TransposeDefect,
)

class UnsupportedHookConversionError(ValueError):
    """A conversion cannot be inverted safely from the captured layout alone."""

class TransformerLensConversionRegistry:
    
    _STEPS = "transformer_lens.conversion_utils.conversion_steps."
    _COMPONENTS = "transformer_lens.model_bridge.generalized_components."
    _ARCHITECTURES = "transformer_lens.model_bridge.supported_architectures."

    # Some TL classes (ReshapeForAttentionHeads) are defined inside methods and cannot be imported directly.
    # Match both module and class name, never projection/hook-name substrings.
    _CONVERSION_KINDS = {
    (_STEPS + "chain_tensor_conversion", "ChainTensorConversion"): "chain",
    (_STEPS + "rearrange_tensor_conversion", "RearrangeTensorConversion"): "rearrange",
    (_STEPS + "transpose_tensor_conversion", "TransposeTensorConversion"): "transpose",
    (_STEPS + "attention_auto_conversion", "AttentionAutoConversion"): "attention_auto",
    (_COMPONENTS + "attention", "ReshapeForAttentionHeads"): "heads",
    (_COMPONENTS + "attention", "TransposeRotaryHeads"): "rotary",
    (_COMPONENTS + "joint_qkv_attention", "ConditionalRearrangeConversion"): "heads",
    (_ARCHITECTURES + "gpt_bigcode", "MQAQKVConversionRule"): "heads",
    (_ARCHITECTURES + "opt", "_UnflattenTokens"): "tokens",
    }
    
    @staticmethod
    def map_conversion_to_defects(
        conversion: BaseTensorConversion | None,
        captured_shape: tuple[int, ...],
    ) -> list[Defect]:
        """Snapshot defects in forward order; restoration applies them in reverse.

        Meta tensors validate inverses and intermediate chain shapes without copying
        cached activations or allocating tensor storage on CPU/GPU.
        """
        if conversion is None:
            return []
        if conversion.input_filter is not None or conversion.output_filter is not None:
            raise UnsupportedHookConversionError("Conversion filters require an explicit inverse rule")

        key = (type(conversion).__module__, type(conversion).__name__)
        kind = TransformerLensConversionRegistry._CONVERSION_KINDS.get(key)
        if kind is None:
            raise UnsupportedHookConversionError(f"No inverse rule for {key[0]}.{key[1]}")

        if kind == "chain":
            defects: list[Defect] = []
            shape = captured_shape
            for child in reversed(getattr(conversion, "conversions")):
                child_defects = TransformerLensConversionRegistry.map_conversion_to_defects(child, shape)
                restored = torch.empty(shape, device="meta")
                for defect in reversed(child_defects):
                    restored = defect.restore_activation(restored)
                shape = tuple(restored.shape)
                defects = child_defects + defects
            return defects

        ndim = len(captured_shape)
        if kind == "attention_auto":
            if ndim == 4:
                raise UnsupportedHookConversionError(
                    "AttentionAutoConversion on 4D is ambiguous without the pre-conversion layout"
                )
            return []

        if kind == "heads":
            if ndim != 4:
                # ReshapeForAttentionHeads leaves mismatched (e.g. TP) widths alone.
                return []
            if key[1] == "ReshapeForAttentionHeads" and captured_shape[-2:] != (
                getattr(conversion, "n_heads"),
                getattr(conversion, "d_head"),
            ):
                raise UnsupportedHookConversionError("Captured head axes do not match the conversion")
            return [ReshapeDefect((*captured_shape[:-2], captured_shape[-2] * captured_shape[-1]))]

        if kind == "rotary":
            return [TransposeDefect((1, 2))] if ndim == 4 else []
        if kind == "transpose":
            return [TransposeDefect((0, 1))] if ndim == 2 else []
        if kind == "tokens":
            batch_seq = getattr(conversion, "batch_seq")
            if ndim == 3 and batch_seq is not None and captured_shape[:2] == tuple(batch_seq):
                return [ReshapeDefect((captured_shape[0] * captured_shape[1], captured_shape[2]))]
            return []

        # Rearrange can include a permutation even when the before/after shapes match.
        pattern = getattr(conversion, "pattern")
        left, right = pattern.split("->")
        defect = RearrangeDefect((
            f"{right.strip()} -> {left.strip()}",
            dict(getattr(conversion, "axes_lengths")),
        ))
        try:
            defect.restore_activation(torch.empty(captured_shape, device="meta"))
        except (ValueError, RuntimeError) as exc:
            raise UnsupportedHookConversionError(
                f"Cannot invert {pattern!r} for {captured_shape}: {exc}"
            ) from exc
        return [defect]
