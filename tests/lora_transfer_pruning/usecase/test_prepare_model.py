from unittest.mock import Mock

import pytest
import torch
from peft import LoraConfig
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch import nn
from transformers import (
    AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, Qwen3MoeConfig, Qwen3MoeForCausalLM,
)

from lora_transfer_pruning.adapter.model_loading.hugging_face_model_loader import HuggingFaceModelLoader
from lora_transfer_pruning.adapter.quantization.torchao_quantizer_int4.torchao_int4_linear import TorchAOInt4Linear
from lora_transfer_pruning.usecase.prepare_model import PrepareModelUseCase


@pytest.mark.parametrize(
    ("model_class", "config_class", "expert_config"),
    [
        (LlamaForCausalLM, LlamaConfig, {}),
        (
            Qwen3MoeForCausalLM,
            Qwen3MoeConfig,
            {"moe_intermediate_size": 32, "num_experts": 4, "num_experts_per_tok": 2},
        ),
    ],
    ids=["llama", "qwen3-moe"],
)
def test_preparation_preserves_trainable_lora_parameters(
    tmp_path, monkeypatch, model_class, config_class, expert_config,
):
    if expert_config:
        pytest.importorskip("defuser")
    prepare = PrepareModelUseCase()

    config = config_class(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=32,
        tie_word_embeddings=True, **expert_config,
    )
    tokens = torch.tensor([[1, 2, 3]])
    original = model_class(config).eval()
    original.save_pretrained(tmp_path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<unk>": 0, "<s>": 1, "</s>": 2}, unk_token="<unk>")),
        unk_token="<unk>", bos_token="<s>", eos_token="</s>", pad_token="<unk>", add_bos_token=True,
    )
    tokenizer.save_pretrained(tmp_path)
    base = prepare.load_model(tmp_path, local_files_only=True)
    torch.testing.assert_close(base(tokens).logits, original(tokens).logits, atol=1e-6, rtol=1e-5)
    if expert_config:
        experts = base.model.layers[0].mlp.experts
        assert isinstance(experts.get_submodule("0.up_proj"), nn.Linear)
        fused = prepare.load_model(tmp_path, local_files_only=True, defuse_experts=False)
        assert isinstance(fused.model.layers[0].mlp.experts.gate_up_proj, nn.Parameter)
    attention = base.model.layers[0].self_attn
    dense_projection = attention.q_proj
    with pytest.raises(ValueError, match="requires CUDA FP16/BF16"):
        prepare.quantize(base, device="cpu")
    assert attention.q_proj is dense_projection

    # Exercise LoRA and the bridge around a real packed INT4 layer on CPU.
    # Its forward uses the dense reference because the fused kernel needs CUDA.
    def reference_forward(layer, x):
        return nn.functional.linear(x, layer.reference_weight().to(x.dtype), layer.bias)

    monkeypatch.setattr(TorchAOInt4Linear, "forward", reference_forward)
    packed_projection = TorchAOInt4Linear.from_packed({
        "qweight": torch.full(
            (dense_projection.in_features // 2, dense_projection.out_features), 0x11, dtype=torch.uint8,
        ),
        "scales": torch.full((1, dense_projection.out_features), 0.125, dtype=torch.float16),
        "zeros": torch.zeros((1, dense_projection.out_features), dtype=torch.float16),
    }, device="cpu")
    attention.q_proj = packed_projection

    targets = ["q_proj", "up_proj"] if expert_config else ["q_proj"]
    lora_config = LoraConfig(r=2, target_modules=targets, task_type="CAUSAL_LM")
    model = prepare.add_lora(base, lora_config)
    assert lora_config.base_model_name_or_path is None
    assert model.peft_config["default"].base_model_name_or_path == str(tmp_path)
    original_grad_flags = {p: p.requires_grad for p in model.parameters()}
    projection = base.model.layers[0].self_attn.q_proj
    assert projection.base_layer is packed_projection
    for adapted in (model, base):
        with pytest.raises(TypeError, match="before adding LoRA"):
            prepare.quantize(adapted, device="cpu")

    bridge = prepare.bridge_model(model)
    assert bridge.original_model is base
    assert prepare.bridge_model(bridge) is bridge
    assert {p: p.requires_grad for p in model.parameters() if p in original_grad_flags} == original_grad_flags

    logits = bridge(input=tokens, use_cache=False, return_type="logits")
    logits.square().mean().backward()
    assert projection.lora_B["default"].weight.grad is not None
    if expert_config:
        assert any(
            parameter.grad is not None
            for name, parameter in base.named_parameters()
            if ".experts." in name and ".up_proj.lora_B." in name
        )


def test_loader_leaves_gptq_expert_layout_to_the_backend(monkeypatch):
    model = LlamaForCausalLM(LlamaConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, vocab_size=32,
    ))
    model.config.quantization_config = {"quant_method": "gptq"}
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", Mock(return_value=model))
    loader = HuggingFaceModelLoader()
    defuse = Mock()
    monkeypatch.setattr(loader, "_defuse", defuse)

    assert loader.load("checkpoint") is model
    defuse.assert_not_called()
