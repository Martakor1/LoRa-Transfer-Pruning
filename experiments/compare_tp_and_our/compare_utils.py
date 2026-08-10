# Fraction-based version of the activation-vs-structural comparison.
# Run on a freshly loaded, unpruned bridge; skip the preceding explicit-index
# comparison cell after restarting the kernel.
import gc
import torch
import torch_pruning as tp
from lora_transfer_pruning.usecase.local_pruning import LocalPruning
from lora_transfer_pruning.usecase.prune_task_type import PruneTaskType
from experiments.utils import evaluate_language_model
from rope_resize_for_tp import make_gemma_rope_resize_pre_hook

def compare_tp_and_transfer_pruning(model_bridge, 
                                    fraction_attn_layers: list[int], 
                                    fraction_mlp_layers: list[int], 
                                    attn_out_fraction: list[int] | float, 
                                    mlp_out_fraction: list[int] | float, 
                                    seed: int, 
                                    evaluation_batches: torch.Tensor,
                                    eval_batch_size: int
                                    ):
    DEVICE = model_bridge.device

    original_head_dim = int(model_bridge.model.config.head_dim)
    expected_q_width = int(model_bridge.model.config.num_attention_heads) * original_head_dim
    for layer in fraction_attn_layers:
        actual_q_width = model_bridge.blocks[layer].attn.q._original_component.out_features
        assert actual_q_width == expected_q_width, (
            "Fraction experiment requires a fresh, unpruned model_bridge. "
            f"Layer {layer}: q.out_features={actual_q_width}, expected={expected_q_width}. "
            "Restart the kernel, run the setup cells, skip the explicit-index pruning cell, "
            "then run this cell."
        )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    fraction_prune_task: PruneTaskType = {
        **{
            f"blocks.{layer}.attn.q": (None, attn_out_fraction)
            for layer in fraction_attn_layers
        },
        **{
            f"blocks.{layer}.mlp.up_proj": (None, mlp_out_fraction)
            for layer in fraction_mlp_layers
        },
    }
    print("fraction prune_task:")
    for module_name, task in fraction_prune_task.items():
        print(f"  {module_name}: {task}")

    model_bridge.reset_hooks()
    fraction_local_pruning = LocalPruning(
        model_bridge,
        evaluation_batches[:1].to(DEVICE),
    )
    fraction_groups = fraction_local_pruning.get_torch_pruning_groups(
        fraction_prune_task
    )
    assert len(fraction_groups) == len(fraction_prune_task)

    # Save every root index set and the local RoPE coordinates for attention.
    fraction_removed_idxs = {}
    fraction_attn_local_idxs = {}
    for (module_name, _), pruning_group in zip(
        fraction_prune_task.items(), fraction_groups
    ):
        root_item = pruning_group[0]
        root_hybrid_idxs = root_item.idxs
        root_flat_idxs = sorted({
            int(idx) for idx in tp._helpers.to_plain_idxs(root_hybrid_idxs)
        })
        fraction_removed_idxs[module_name] = root_flat_idxs

        if ".attn.q" not in module_name:
            continue
        layer = int(module_name.split(".")[1])
        fraction_attn_local_idxs[layer] = torch.tensor(
            sorted({idx % original_head_dim for idx in root_flat_idxs}),
            dtype=torch.long,
        )

    print("removed root indices by module:")
    for module_name, removed_idxs in fraction_removed_idxs.items():
        print(
            f"  {module_name}: count={len(removed_idxs)}, idxs={removed_idxs}"
        )

    for layer, local_idxs in fraction_attn_local_idxs.items():
        paired = torch.where(
            local_idxs < original_head_dim // 2,
            local_idxs + original_head_dim // 2,
            local_idxs - original_head_dim // 2,
        )
        assert set(local_idxs.tolist()) == set(paired.tolist())
        print(
            f"layer={layer}: removed local head dims={len(local_idxs)} "
            f"({len(local_idxs) / original_head_dim:.3%}), idxs={local_idxs.tolist()}"
        )

    fraction_baseline_metrics = evaluate_language_model(
        model_bridge, evaluation_batches, batch_size=eval_batch_size
    )

    fraction_local_pruning.prepare_model_to_transfer_pruning_from_groups(fraction_groups)
    fraction_transfer_metrics = evaluate_language_model(
        model_bridge, evaluation_batches, batch_size=eval_batch_size
    )

    model_bridge.reset_hooks()
    for pruning_group in fraction_groups:
        pruning_group.prune()

    # Groups retain DependencyGraph's autograd trace.
    del fraction_groups, fraction_local_pruning, pruning_group 
    globals().pop("DG", None)
    globals().pop("group", None)
    globals().pop("group_kv", None)
    gc.collect()
    torch.cuda.empty_cache()


    #-------manual model changing for TP, cos sin hooks for Rope and saving shapes--------
    #deprecated, not all models contain attn.hook_cos
    # def make_fraction_compact_rope_hook(keep_idxs):
    #     def compact_rope_hook(activation, hook):
    #         return activation.index_select(-1, keep_idxs.to(activation.device))
    #     return compact_rope_hook

    fraction_structural_shapes = {}
    for layer, local_idxs in fraction_attn_local_idxs.items():
        removed = set(local_idxs.tolist())
        keep_idxs = torch.tensor(
            [idx for idx in range(original_head_dim) if idx not in removed],
            dtype=torch.long,
        )
        attn = model_bridge.blocks[layer].attn
        attn._original_component.head_dim = len(keep_idxs)
        
        attn.register_forward_pre_hook( #no need to unregister
            make_gemma_rope_resize_pre_hook(keep_idxs),
            with_kwargs=True,
        )
        fraction_structural_shapes[f"blocks.{layer}.attn"] = {
            "q": tuple(attn.q._original_component.weight.shape),
            "k": tuple(attn.k._original_component.weight.shape),
            "v": tuple(attn.v._original_component.weight.shape),
            "o": tuple(attn.o._original_component.weight.shape),
            "head_dim": attn._original_component.head_dim,
        }
    #----------------------------------------   

    for layer in fraction_mlp_layers:
        mlp = model_bridge.blocks[layer].mlp
        fraction_structural_shapes[f"blocks.{layer}.mlp"] = {
            "up": tuple(mlp.up_proj._original_component.weight.shape),
            "gate": tuple(mlp.gate_proj._original_component.weight.shape),
            "out": tuple(mlp.out._original_component.weight.shape),
        }

    print("fraction structural shapes:", fraction_structural_shapes)
    fraction_structural_metrics = evaluate_language_model(
        model_bridge, evaluation_batches, batch_size=eval_batch_size
    )

    fraction_comparison = {
        "baseline": fraction_baseline_metrics,
        "transfer_activation": fraction_transfer_metrics,
        "torch_pruning_structural": fraction_structural_metrics,
        "structural_minus_transfer": {
            key: fraction_structural_metrics[key] - fraction_transfer_metrics[key]
            for key in ("loss", "perplexity")
        },
    }
    return fraction_comparison