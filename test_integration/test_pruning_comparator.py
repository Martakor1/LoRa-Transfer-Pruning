"""Real-checkpoint equivalents of test_llama_pruning_comparator.ipynb.

Run serially: each comparison mutates its fresh model into a TP model.
Heavy imports happen after the opt-in fixture, so collection needs no CUDA.

From the repository root, for example:
    HF_HUB_CACHE=/glazkov-dev/.cache HF_HUB_OFFLINE=1 .venv/bin/python -m pytest \
        test_integration -k llama --run-pruning-integration --pruning-device=cuda:3 -s

Each family runs in a separate pytest subprocess: failure tracebacks must not
retain a whole GPU model and make the next architecture fail with OOM.
Without --pruning-allow-download, checkpoints must already be cached.

Coverage: real checkpoint, one multi-token prompt, forward/backward and loss;
no training, quantization, LoRA, DeepSeek experts or Gemma shared-KV pruning.
These are numerical regression tests, not a proof for every possible input.
"""

import os
from pathlib import Path
import subprocess
import sys

import pytest


CASES = [
    pytest.param("meta-llama/Llama-3.1-8B-Instruct", "llama", "float16", id="llama"),
    pytest.param("google/gemma-4-E4B-it", "gemma", "float16", id="gemma"),
    pytest.param("deepseek-ai/DeepSeek-V2-Lite-Chat", "deepseek", "float16", id="deepseek"),
]

# FP16 GEMM shape changes and the SP RMSNorm correction are not bitwise exact.
# FP16 is intentional: BF16 accumulated up to 9% gradient relative-L2 error
# in the Llama checkpoint. Use --pruning-dtype=bfloat16 to reproduce that test,
# rather than silently widening tolerances until it passes.
# The relative-L2 guard below additionally prevents this absolute tolerance
# from accepting missing/zeroed small gradients.
ATOL = 5e-2
RTOL = 5e-2
MAX_REL_L2 = 5e-2


def make_prune_task(bridge, family):
    from lora_transfer_pruning.core.prune_task_type import GroupPruneTask, ModelPruneTask

    count = len(bridge.blocks)
    assert count >= 4
    layers = [0, count // 2, count - 1]
    if family == "gemma":
        # E4B: later blocks share KV, and full attention has a different head
        # width. Match the notebook's sliding-attention path without pruning
        # the same shared-KV dependency group more than once.
        layers = [0, 6, 12]
        assert count > 20
        for layer in layers:
            attn = bridge.blocks[layer].attn._original_component
            assert not attn.is_kv_shared_layer
            assert attn.head_dim == 256
    tasks = {}
    for layer in layers:
        if family == "deepseek":
            # Exercise both non-rotary coordinates and complex RoPE pairs.
            nope_dim = bridge.blocks[layer].attn.qk_nope_head_dim
            indices = [1, 2, nope_dim + 2]
        else:
            indices = [1, 2, 99] if family == "gemma" else [1, 2, 9]
        tasks[f"blocks.{layer}.attn.q_proj"] = GroupPruneTask(cols=None, rows=indices)

    # DeepSeek's dense/shared expert bridge is not yet supported by this API
    # (see test_pruning_task_dict_deepseek.ipynb); do not bypass the builder.
    if family != "deepseek":
        mlp_layers = [4, 20] if family == "gemma" else [1, count // 2 + 1]
        for layer in mlp_layers:
            tasks[f"blocks.{layer}.mlp.up_proj"] = GroupPruneTask(
                cols=None, rows=[3, 5, 1000],
            )
    return ModelPruneTask(tasks)


def select_linear_hooks(bridge, task, family):
    """Use canonical HookPoint names, including native upstream linear inputs."""
    names = set()
    for path in task.data:
        parent_path = path.rsplit(".", 1)[0]
        parent = bridge.get_submodule(parent_path)
        if ".attn." in path:
            components = ("q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj") if family == "deepseek" else ("q", "k", "v", "o")
        else:
            components = ("up_proj", "gate_proj", "down_proj")
        for component in components:
            module = getattr(parent, component)
            for hook in (module.hook_in, module.hook_out):
                assert hook.name in bridge.hook_dict, (path, component, hook.name)
                names.add(hook.name)
    return sorted(names)


def assert_comparison_rows(rows, expected_names, label):
    from pruning_lens.core.compairing_trace import ComparisonStatus

    assert set(rows) == set(expected_names), f"{label}: missing or unexpected hooks"
    failures = []
    for name, row in rows.items():
        if (row.status != ComparisonStatus.OK or row.compared_elements <= 0
                or row.rel_l2 is None or row.rel_l2 > MAX_REL_L2):
            failures.append(
                f"{name}: {row.status}, max={row.max_abs}, rmse={row.rmse}, "
                f"rel_l2={row.rel_l2}, shapes={row.reference_restored_shape}/"
                f"{row.candidate_restored_shape}, reason={row.reason}"
            )
    assert not failures, label + "\n" + "\n".join(failures)


@pytest.mark.parametrize("model_id,family,dtype_name", CASES)
def test_pretrained_transfer_matches_structural_pruning(
    request, model_id, family, dtype_name,
):
    if not request.config.getoption("--run-pruning-integration"):
        pytest.skip("Use --run-pruning-integration to load real LLM checkpoints")
    if os.environ.get("PRUNING_INTEGRATION_CHILD") != "1":
        command = [
            sys.executable, "-m", "pytest", f"{Path(__file__).resolve()}::{request.node.name}",
            "--run-pruning-integration", "--pruning-device=" + request.config.getoption("--pruning-device"),
            "-q", "-s", "--tb=short", "--disable-warnings",
        ]
        if request.config.getoption("--pruning-allow-download"):
            command.append("--pruning-allow-download")
        if request.config.getoption("--pruning-dtype"):
            command.append("--pruning-dtype=" + request.config.getoption("--pruning-dtype"))
        completed = subprocess.run(
            command, cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "PRUNING_INTEGRATION_CHILD": "1"},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=1200,
        )
        print(completed.stdout)
        assert completed.returncode == 0, f"{family} integration failed:\n{completed.stdout}"
        return

    pruning_device = request.getfixturevalue("pruning_device")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformer_lens.model_bridge import TransformerBridge
    from pruning_lens.core.compairing_trace import ComparisonStatus
    from pruning_lens.usecase.compare_pruned_models_usecase import ComparePrunedModelsUsecase

    dtype = getattr(torch, request.config.getoption("--pruning-dtype") or dtype_name)
    local_only = not request.config.getoption("--pruning-allow-download")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=False, local_files_only=local_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=False, local_files_only=local_only,
        device_map={"": str(pruning_device)}, dtype=dtype,
        attn_implementation="eager",
    ).eval()
    model.config.use_cache = False
    model.config.get_text_config().use_cache = False
    if family == "deepseek":
        # Same router assumptions as test_pruning_task_dict_deepseek.ipynb.
        router = model.model.layers[1].mlp.gate
        assert isinstance(router, torch.nn.Linear), type(router)
        assert tuple(router.weight.shape) == (model.config.n_routed_experts, model.config.hidden_size)
    bridge = TransformerBridge.boot_transformers(model_id, hf_model=model, dtype=dtype)
    try:
        # More than one token is essential: single-token softmax can hide Q/K errors.
        tokens = tokenizer(
            "The attention mechanism allows a language model to use information "
            "from earlier tokens. Pruning should preserve the retained channels.",
            return_tensors="pt",
        ).input_ids.to(pruning_device)
        assert tokens.shape[-1] > 8
        task = make_prune_task(bridge, family)
        names = select_linear_hooks(bridge, task, family)
        before = {
            path: tuple(bridge.get_submodule(path)._original_component.weight.shape)
            for path in task.data
        }
        result, sp_trace, tp_trace = ComparePrunedModelsUsecase.compare_transfer_and_torch_prunings(
            bridge, tokens, task, names_filter=names, compute_gradient=True,
            atol=ATOL, rtol=RTOL, store_difference=False, print_summary=True,
        )
        for trace in (sp_trace, tp_trace):
            assert torch.isfinite(trace.metric).all()
        print(
            f"{family}: SP loss={sp_trace.metric.item():.6f}, "
            f"TP loss={tp_trace.metric.item():.6f}; "
            f"SP perplexity={sp_trace.metric.double().exp().item():.6f}, "
            f"TP perplexity={tp_trace.metric.double().exp().item():.6f}"
        )
        assert_comparison_rows(result.activations, names, "FWD")
        assert_comparison_rows(result.gradients, names, "BWD")
        assert result.metric is not None and result.metric.status == ComparisonStatus.OK
        for trace in (sp_trace, tp_trace):
            assert torch.isfinite(trace.metric).all()
            assert any(g is not None and torch.count_nonzero(g) > 0 for g in trace.gradients.values())
        # A no-op pruning implementation must not make this test pass.
        for path, old_shape in before.items():
            linear = bridge.get_submodule(path)._original_component
            assert linear.weight.shape[0] < old_shape[0], path
            assert tuple(linear.weight.shape) == (linear.out_features, linear.in_features), path
    finally:
        bridge.reset_hooks()
