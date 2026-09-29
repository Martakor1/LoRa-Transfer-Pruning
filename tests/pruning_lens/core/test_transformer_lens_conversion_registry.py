"""Tests for pruning_lens.core.transformer_lens_conversion_registry."""

from types import SimpleNamespace

import pytest
import torch
from transformer_lens.conversion_utils.conversion_steps.attention_auto_conversion import AttentionAutoConversion
from transformer_lens.conversion_utils.conversion_steps.base_tensor_conversion import BaseTensorConversion
from transformer_lens.conversion_utils.conversion_steps.chain_tensor_conversion import ChainTensorConversion
from transformer_lens.conversion_utils.conversion_steps.rearrange_tensor_conversion import RearrangeTensorConversion
from transformer_lens.conversion_utils.conversion_steps.transpose_tensor_conversion import TransposeTensorConversion
from transformer_lens.model_bridge.generalized_components.joint_qkv_attention import JointQKVAttentionBridge
from transformer_lens.model_bridge.supported_architectures.gpt_bigcode import MQAQKVConversionRule
from transformer_lens.model_bridge.supported_architectures.opt import _UnflattenTokens

from pruning_lens.core.activation_corruption import ActivationCorruption
from pruning_lens.core.transformer_lens_conversion_registry import (
    UnsupportedHookConversionError,
    TransformerLensConversionRegistry,
)


@pytest.mark.parametrize("projection,width", [("q", 6), ("k", 3), ("v", 3), ("o", 6)])
def test_projection_heads_roundtrip(projection, width, attention_block_with_conversions):
    attn = attention_block_with_conversions
    hook = getattr(attn, projection).hook_in if projection == "o" else getattr(attn, projection).hook_out
    source = torch.arange(2 * 5 * width).reshape(2, 5, width)
    converted = hook.hook_conversion.convert(source)
    assert converted.ndim == 4
    corruption = ActivationCorruption(TransformerLensConversionRegistry.map_conversion_to_defects(hook.hook_conversion, tuple(converted.shape)))
    assert torch.equal(corruption.restore_activation(converted), source)


def test_mismatched_projection_width_has_no_defect(attention_block_with_conversions):
    conversion = attention_block_with_conversions.q.hook_out.hook_conversion
    source = torch.randn(2, 5, 4)
    assert conversion.convert(source) is source
    assert TransformerLensConversionRegistry.map_conversion_to_defects(conversion, tuple(source.shape)) == []


@pytest.mark.parametrize("shape", [(2, 2, 5, 3), (2, 2, 2, 3)])
def test_rotary_transpose_including_equal_axis_sizes(shape, attention_block_with_conversions):
    conversion = attention_block_with_conversions.hook_rot_q.hook_conversion
    source = torch.arange(torch.tensor(shape).prod()).reshape(shape)
    converted = conversion.convert(source)
    corruption = ActivationCorruption(TransformerLensConversionRegistry.map_conversion_to_defects(conversion, tuple(converted.shape)))
    assert torch.equal(corruption.restore_activation(converted), source)


@pytest.mark.parametrize("shape", [(2, 5, 6), (2, 5, 3)])
def test_mqa_heads_roundtrip(shape):
    conversion = MQAQKVConversionRule(n_heads=2, d_head=3)
    source = torch.randn(shape)
    converted = conversion.convert(source)
    corruption = ActivationCorruption(TransformerLensConversionRegistry.map_conversion_to_defects(conversion, tuple(converted.shape)))
    assert torch.equal(corruption.restore_activation(converted), source)


def test_joint_qkv_heads_roundtrip():
    conversion = JointQKVAttentionBridge._create_qkv_conversion_rule(
        SimpleNamespace(config=SimpleNamespace(n_heads=2))
    )
    source = torch.randn(2, 5, 6)
    converted = conversion.convert(source)
    corruption = ActivationCorruption(TransformerLensConversionRegistry.map_conversion_to_defects(conversion, tuple(converted.shape)))
    assert torch.equal(corruption.restore_activation(converted), source)


def test_opt_tokens_snapshot():
    conversion = _UnflattenTokens()
    conversion.batch_seq = (2, 5)
    source = torch.randn(10, 6)
    converted = conversion.convert(source)
    corruption = ActivationCorruption(TransformerLensConversionRegistry.map_conversion_to_defects(conversion, tuple(converted.shape)))
    conversion.batch_seq = (3, 7)
    assert torch.equal(corruption.restore_activation(converted), source)


def test_nested_chain_roundtrip_and_metadata_snapshot():
    rearrange = RearrangeTensorConversion("b s (h d) -> b s h d", h=2)
    conversion = ChainTensorConversion([
        rearrange,
        ChainTensorConversion([RearrangeTensorConversion("b s h d -> b h s d")]),
    ])
    source = torch.arange(60).reshape(2, 5, 6)
    converted = conversion.convert(source)
    corruption = ActivationCorruption(TransformerLensConversionRegistry.map_conversion_to_defects(conversion, tuple(converted.shape)))
    rearrange.axes_lengths["h"] = 99
    assert len(corruption.defects) == 2
    assert torch.equal(corruption.restore_activation(converted), source)


def test_same_shape_rearrange_is_not_identity():
    conversion = RearrangeTensorConversion("b s h d -> b h s d")
    source = torch.arange(24).reshape(2, 2, 2, 3)
    converted = conversion.convert(source)
    assert converted.shape == source.shape
    assert not torch.equal(converted, source)
    corruption = ActivationCorruption(TransformerLensConversionRegistry.map_conversion_to_defects(conversion, tuple(converted.shape)))
    assert torch.equal(corruption.restore_activation(converted), source)


def test_rearrange_missing_axis_lengths_is_rejected():
    conversion = RearrangeTensorConversion("b h d -> b (h d)")
    with pytest.raises(UnsupportedHookConversionError, match="Cannot invert"):
        TransformerLensConversionRegistry.map_conversion_to_defects(conversion, (2, 6))


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4)])
def test_transpose_roundtrip_or_noop(shape):
    conversion = TransposeTensorConversion()
    source = torch.randn(shape)
    converted = conversion.convert(source)
    corruption = ActivationCorruption(TransformerLensConversionRegistry.map_conversion_to_defects(conversion, tuple(converted.shape)))
    assert torch.equal(corruption.restore_activation(converted), source)
    assert bool(corruption.defects) == (len(shape) == 2)


def test_attention_auto_rejects_4d_only():
    conversion = AttentionAutoConversion(SimpleNamespace(n_heads=2))
    assert TransformerLensConversionRegistry.map_conversion_to_defects(conversion, (2, 5, 6)) == []
    with pytest.raises(UnsupportedHookConversionError, match="4D is ambiguous"):
        TransformerLensConversionRegistry.map_conversion_to_defects(conversion, (2, 2, 5, 5))


def test_unknown_conversion_and_filters_are_rejected():
    with pytest.raises(UnsupportedHookConversionError, match="No inverse rule"):
        TransformerLensConversionRegistry.map_conversion_to_defects(BaseTensorConversion(), (2, 3))
    conversion = RearrangeTensorConversion("a b -> b a", input_filter=lambda x: x)
    with pytest.raises(UnsupportedHookConversionError, match="filters"):
        TransformerLensConversionRegistry.map_conversion_to_defects(conversion, (2, 3))
