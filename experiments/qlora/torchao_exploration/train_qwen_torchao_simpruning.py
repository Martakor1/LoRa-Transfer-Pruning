"""TorchAO + TL + LocalPruning, then the same SFT experiment as the baseline."""

import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.qlora.torchao_exploration.train_qwen_torchao_sft import ExperimentConfig, run_experiment, plot_training_history
from experiments.qlora.torchao_exploration.train_qwen_torchao import GPTQ_MODEL, HERE
from experiments.qlora.torchao_exploration.torchao_int4 import TorchAOInt4Linear
import torch
import torch_pruning as tp
from transformer_lens.model_bridge.transformer_bridge import TransformerBridge
from transformer_lens.factories.architecture_adapter_factory import ArchitectureAdapterFactory
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge
from lora_transfer_pruning.core.tl_adapters import Qwen3MoeWithExpertBridges
from lora_transfer_pruning.core.prune_task_type import GroupPruneTask, ModelPruneTask
from lora_transfer_pruning.usecase.local_pruning import LocalPruning
from lora_transfer_pruning.adapter.torch_pruning.tp_utils import iter_pruned_linear_hooks

OUTPUT = HERE / "slop_experiments" / "6_torchao_simpruning_train" / "torchao_sft_simpruning"
LAYERS = [45, 46, 47]
ATTN_RATIO = 0.3
MLP_RATIO = 0.3
PRUNING_SEED = 0


class TorchAOGraphOnlyPruner(tp.pruner.function.BasePruningFunc):
    """Keep Linear input/output axes separate in DG; never slice packed weights."""
    def get_in_channels(self, layer):
        return layer.in_features

    def get_out_channels(self, layer):
        return layer.out_features

    def prune_in_channels(self, layer, idxs):
        raise RuntimeError("TorchAO INT4: graph-only registration, use sim-pruning")

    def prune_out_channels(self, layer, idxs):
        raise RuntimeError("TorchAO INT4: graph-only registration, use sim-pruning")


def prepare_simpruning(model, tokenizer, train, trainer, output):
    started = time.perf_counter()
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    ArchitectureAdapterFactory.register_adapter("Qwen3MoeForCausalLM", Qwen3MoeWithExpertBridges)
    bridge = TransformerBridge.boot_transformers(
        GPTQ_MODEL, hf_model=model.get_base_model(), dtype=torch.float16,
    )
    assert bridge.original_model is model.get_base_model()
    assert isinstance(bridge.blocks[47].mlp.experts[0].up_proj, LinearBridge)
    # TL may add parameters (e.g. a zero unembedding bias). They are not LoRA
    # and must not become trainable or enter our existing optimizer.
    assert trainable_ids <= {id(p) for p in model.parameters()}, "TL replaced adapter parameters"
    newly_trainable = [name for name, p in model.named_parameters()
                       if p.requires_grad and id(p) not in trainable_ids]
    print("Freezing additional TL parameters:", newly_trainable, flush=True)
    for p in model.parameters():
        p.requires_grad_(id(p) in trainable_ids)
    print("TL wrapping complete; evaluating before masks", flush=True)
    trainer.evaluate()
    wrapped_before = dict(trainer.state.log_history[-1])
    task = ModelPruneTask({
        **{f"blocks.{layer}.attn.q": GroupPruneTask(None, ATTN_RATIO) for layer in LAYERS},
        **{f"blocks.{layer}.mlp.experts.{expert}.up_proj": GroupPruneTask(None, MLP_RATIO)
           for layer in LAYERS for expert in range(len(bridge.blocks[layer].mlp.experts))},
    }, name="default")
    tokens = torch.tensor(train[0]["input_ids"][:16], device="cuda")[None]
    print("Building DG, including synthetic expert forwards", flush=True)
    local = LocalPruning(bridge, tokens, peft_model=model,
                         customized_pruners={TorchAOInt4Linear: TorchAOGraphOnlyPruner()})
    torch.manual_seed(PRUNING_SEED)
    print("Building pruning groups", flush=True)
    plan = local.get_torch_pruning_groups_and_structural_setups(task)
    boundaries = list(iter_pruned_linear_hooks(bridge, plan.groups))
    manifest = [{"hook": hook.name, "removed_indices": list(map(int, idxs))}
                for hook, idxs in boundaries]
    (output / "pruning_plan.json").write_text(json.dumps(manifest, indent=2))
    print(f"Preparing {len(plan.groups)} groups; {len(boundaries)} boundaries", flush=True)
    local.prepare_model_to_transfer_pruning([plan], rescale=False)
    # The SFTTrainer still calls PEFT -> HF; TL replaced the actual HF children
    # in place. Prove those hooks fire on THAT path, not only bridge.forward().
    audit = audit_masks(model, bridge, task, boundaries, tokens)
    assert {id(p) for p in model.parameters() if p.requires_grad} == trainable_ids
    result = {"enabled": True, "layers": LAYERS, "attention_ratio_requested": ATTN_RATIO,
              "mlp_ratio_requested": MLP_RATIO, "seed": PRUNING_SEED, "rescale": False,
              "group_count": len(plan.groups), "linear_boundary_count": len(boundaries),
              "wrapped_unpruned_eval": wrapped_before, "mask_audit": audit,
              "frozen_additional_tl_parameters": newly_trainable,
              "attention_implementation": bridge.config._attn_implementation,
              "preparation_seconds": time.perf_counter() - started}
    # Do NOT execute structural setups or group.prune(): SP keeps all shapes.
    # Drop DG's saved graph/CPU tensors; masks only need their channel indices.
    del boundaries, plan, local
    gc.collect()
    torch.cuda.empty_cache()
    return result


def audit_masks(model, bridge, task, boundaries, tokens):
    root_hooks = {bridge.get_submodule(name).hook_out.name for name in task.data}
    indices_by_hook = {hook.name: idxs for hook, idxs in boundaries}
    assert root_hooks <= set(indices_by_hook), "Some requested modules have no group indices"
    seen = {}
    handles = []
    for hook, idxs in boundaries:
        if hook.name not in root_hooks:
            continue
        assert len(hook.fwd_hooks) > 0, f"No mask registered: {hook.name}"
        ids = torch.tensor(idxs, device="cuda", dtype=torch.long)

        def capture(tensor, point, ids=ids):
            flat = tensor.flatten(-2) if tensor.ndim == 4 else tensor
            maximum = flat.index_select(-1, ids).abs().max().item()
            seen[point.name] = max(seen.get(point.name, 0.0), maximum)
            return tensor

        # TL hooks use the keyword argument 'hook'. Keep a bound closure.
        def inspect(tensor, hook, fn=capture):
            return fn(tensor, hook)
        handles.append(hook.register_forward_hook(lambda module, args, out, fn=inspect: fn(out, module)))
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss
            assert torch.isfinite(loss)
            for layer in LAYERS:
                for expert in bridge.blocks[layer].mlp.experts:
                    expert(torch.ones(1, bridge.config.hidden_size, device="cuda", dtype=torch.float16))
        assert root_hooks <= set(seen), "Requested hooks did not fire"
        assert all(value == 0 for value in seen.values()), seen
    finally:
        for handle in handles:
            handle.remove()
    print(f"Verified {len(seen)} root outputs: removed coordinates are zero", flush=True)
    return {"checked_root_hooks": len(seen), "max_removed_activation": max(seen.values()),
            "hf_peft_probe_loss": loss.item()}


if __name__ == "__main__":
    result = run_experiment(ExperimentConfig(), output=OUTPUT, prepare_model=prepare_simpruning)
    import matplotlib
    matplotlib.use("Agg")
    plot_training_history(result["history"]).savefig(OUTPUT / "training_history.png", dpi=130)
