"""Shared GPTQ-to-TorchAO loader and LoRA setup for Qwen experiments."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("HF_HUB_CACHE", "/glazkov-dev/.cache")
os.environ.setdefault("HF_DATASETS_CACHE", "/glazkov-dev/.cache/datasets")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
# PEFT imports installed GPTQModel during dispatch. Import it before TorchAO
# creates Autotuners: GPTQModel 7 globally patches their constructor and run().
import gptqmodel  # noqa: F401
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from datasets import load_dataset
from defuser import replace_fused_blocks
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import Linear as PeftLinear
from safetensors import safe_open
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.qlora.torchao_exploration.torchao_int4 import TorchAOInt4Linear, use_fixed_kernel_config
from torchao.prototype.hqq import pack_2xint4

HERE = Path(__file__).resolve().parents[1]  # Shared qlora artifact root.
TORCHAO_EXPERIMENT = HERE / "slop_experiments" / "4_torchao_train"
GPTQ_MODEL = "Qwen/Qwen3-30B-A3B-GPTQ-Int4"
COMPUTE_DTYPE = torch.float16 if os.environ.get("COMPUTE_DTYPE") == "fp16" else torch.bfloat16


def load_gptq_model():
    """Reuse existing GPTQ integers, repack for TorchAO; no requantization.

    Only supports the explicitly verified 4-bit / group-128 / no-act-order
    GPTQ v1 checkpoint. Scales are cast to the chosen compute dtype.
    """
    use_fixed_kernel_config()
    snapshot = Path(snapshot_download(GPTQ_MODEL, local_files_only=True))
    config = AutoConfig.from_pretrained(snapshot)
    quantization = config.quantization_config
    assert quantization["bits"] == 4 and quantization["group_size"] == 128
    assert not quantization["desc_act"] and quantization["checkpoint_format"] == "gptq"
    del config.quantization_config
    config.use_cache = False
    replace_fused_blocks(config.model_type)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, dtype=COMPUTE_DTYPE, attn_implementation="sdpa")
    shifts = torch.arange(0, 32, 4, device="cuda", dtype=torch.int32)
    count = 0
    with safe_open(snapshot / "model.safetensors", framework="pt", device="cpu") as f:
        for name, module in list(model.named_modules()):
            if not isinstance(module, torch.nn.Linear) or name == "lm_head":
                continue
            packed = f.get_tensor(name + ".qweight").cuda()
            q = ((packed[:, None, :] >> shifts[None, :, None]) & 15).flatten(0, 1).to(torch.uint8)
            zeros = f.get_tensor(name + ".qzeros").cuda()
            zeros = (((zeros[:, :, None] >> shifts) & 15) + 1).flatten(1)
            scales = f.get_tensor(name + ".scales").to(device="cuda", dtype=COMPUTE_DTYPE)
            groups = f.get_tensor(name + ".g_idx")
            assert torch.equal(groups, torch.arange(module.in_features) // 128), name
            tensors = {"qweight": pack_2xint4(q.T.contiguous().T),
                       "scales": scales.T.contiguous().T,
                       "zeros": zeros.to(COMPUTE_DTYPE).T.contiguous().T}
            model.set_submodule(name, TorchAOInt4Linear.from_packed(tensors))
            count += 1
            if count % 1000 == 0:
                print(f"Repacked {count} linears; GPU allocated {torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)
        for name, param in list(model.named_parameters()):
            set_module_tensor_to_device(model, name, "cuda", value=f.get_tensor(name).to(COMPUTE_DTYPE))
    assert not any(p.is_meta for p in model.parameters())
    model.to("cuda").requires_grad_(False)
    print(f"Ready: {count} INT4 linears", flush=True)
    return model, AutoTokenizer.from_pretrained(snapshot), snapshot.name


def add_lora(model, adapter_name="default"):
    config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        layers_to_transform=[45, 46, 47], target_parameters=[],
    )
    config._register_custom_module({TorchAOInt4Linear: PeftLinear})
    # The model is explicitly unfused: do not convert expert module targets into parameters.
    with patch("peft.utils.transformers_weight_conversion.convert_peft_config_for_transformers", return_value=None):
        model = get_peft_model(model, config, adapter_name=adapter_name)
    model.print_trainable_parameters()
    return model
