"""Shared CPU model and conversion fixtures for PruningLens tests."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from peft import LoraConfig
from peft.tuners.lora.layer import Linear as LoraLinear
from transformer_lens.HookedRootModule import HookedRootModule
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.attention import AttentionBridge


@pytest.fixture
def attention_block_with_conversions():
    # Instantiate TL's actual nested conversion classes without loading a model.
    attn = SimpleNamespace(
        config=SimpleNamespace(n_heads=2, d_head=3, n_key_value_heads=1),
        hook_aliases={},
        q=SimpleNamespace(hook_out=HookPoint()),
        k=SimpleNamespace(hook_out=HookPoint()),
        v=SimpleNamespace(hook_out=HookPoint()),
        o=SimpleNamespace(hook_in=HookPoint()),
        hook_rot_q=HookPoint(),
        hook_rot_k=HookPoint(),
    )
    AttentionBridge._setup_qkv_hook_reshaping(attn)
    return attn


class _TinyPruningModel(HookedRootModule):
    '''Small CPU model exposing the bridge API used by LocalPruning and capture.'''

    def __init__(self, lora=False):
        super().__init__()
        for name in ("stem", "first", "last"):
            component = LinearBridge(name=name)
            original = nn.Linear(6, 6, bias=False)
            if lora and name == "first":
                original = LoraLinear(
                    original, "default", LoraConfig(r=2, lora_alpha=2), r=2, lora_alpha=2,
                )
                nn.init.normal_(original.lora_B["default"].weight)
            component.set_original_component(original)
            setattr(self, name, component)
        self.setup()

    @property
    def original_model(self):
        return self

    @property
    def model(self):
        return self

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input, **kwargs):
        return self.last(self.first(self.stem(input)))

    def loss_fn(self, logits, tokens):
        return logits.square().mean()


@pytest.fixture
def tiny_pruning_model():
    return _TinyPruningModel
