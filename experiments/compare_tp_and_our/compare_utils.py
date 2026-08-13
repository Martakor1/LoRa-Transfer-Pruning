# Fraction-based version of the activation-vs-structural comparison.
# Run on a freshly loaded, unpruned bridge; skip the preceding explicit-index
# comparison cell after restarting the kernel.
import gc
import torch
from torch import nn
import torch_pruning as tp
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2Model
from lora_transfer_pruning.usecase.local_pruning import LocalPruning
from lora_transfer_pruning.core.prune_task_type import GroupPruneTask, ModelPruneTask
from experiments.utils import evaluate_language_model
from rope_resize_for_tp import make_gemma_rope_resize_pre_hook
from collections import OrderedDict, defaultdict
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.bridge import TransformerBridge
from lora_transfer_pruning.adapter.torch_pruning_group_builder import TorchPruningGroupBuilder


def create_prune_task(fraction_attn_layers: list[int],
                      fraction_mlp_layers: list[int],
                      attn_out_fraction: list[int] | float, 
                      mlp_out_fraction: list[int] | float,
                      q_proj_name = "q",
                      mlp_up_proj_name = "up_proj" 
                      ) -> ModelPruneTask:
    fraction_prune_task: ModelPruneTask = {
            **{
                f"blocks.{layer}.attn.{q_proj_name}": GroupPruneTask(None, attn_out_fraction)
                for layer in fraction_attn_layers
            },
            **{
                f"blocks.{layer}.mlp.{mlp_up_proj_name}": GroupPruneTask(None, mlp_out_fraction)
                for layer in fraction_mlp_layers
            },
        }
    return fraction_prune_task

def prepare_model_for_tp_or_transfer_pruning(
    model_bridge,
    prune_task: ModelPruneTask,
    seed: int,
    evaluation_batches: torch.Tensor,
):
    DEVICE = model_bridge.device
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("fraction prune_task:")
    for module_name, task in prune_task.items():
        print(f"  {module_name}: {task}")

    model_bridge.reset_hooks()
    local_pruning = LocalPruning(
        model_bridge,
        evaluation_batches[:1].to(DEVICE),
    )
    groups = local_pruning.get_torch_pruning_groups(prune_task)

    removed_idxs_by_module = {}
    structural_setups = []
    structural_shape_modules = {}

    # Read the final, corrected indices from groups. In particular, do not
    # reconstruct DeepSeek partial-RoPE mappings from prune_task.
    for module_name, task in prune_task.items():
        module_bridge = model_bridge.get_submodule(module_name)
        module = module_bridge._original_component
        out_idxs = _group_linear_indices(groups, module, prune_out=True)
        removed_idxs_by_module[module_name] = out_idxs.tolist()

        if ".attn." not in module_name or not module_name.endswith(("q", "q_proj")):
            continue

        layer = int(module_name.split(".")[1])
        attn = model_bridge.blocks[layer].attn
        hf_attn = attn._original_component
        is_deepseek_mla = all(
            hasattr(hf_attn, name)
            for name in (
                "q_proj", "kv_a_proj_with_mqa", "kv_b_proj",
                "qk_nope_head_dim", "qk_rope_head_dim", "kv_lora_rank",
            )
        )

        if is_deepseek_mla:
            old_nope = int(hf_attn.qk_nope_head_dim)
            old_rope = int(hf_attn.qk_rope_head_dim)
            old_qk = old_nope + old_rope
            num_heads = int(hf_attn.num_heads)

            q_flat = out_idxs
            q_local = torch.unique(q_flat.remainder(old_qk), sorted=True)
            nope_local = q_local[q_local < old_nope]
            rope_local = q_local[q_local >= old_nope] - old_nope
            closed_rope = TorchPruningGroupBuilder.close_complex_rope_pairs(
                rope_local
            )
            if not torch.equal(closed_rope.cpu(), rope_local.cpu()):
                raise ValueError(
                    f"Layer {layer}: Q-RoPE group indices are not closed over "
                    "DeepSeek complex pairs."
                )
            if len(q_flat) != num_heads * len(q_local):
                raise ValueError(
                    f"Layer {layer}: DeepSeek Q indices are not identical across heads."
                )
            if len(rope_local) % 2:
                raise ValueError(
                    f"Layer {layer}: odd number of real RoPE dimensions removed."
                )

            kv_a_out = _group_linear_indices(
                groups, hf_attn.kv_a_proj_with_mqa, prune_out=True
            )
            kv_b_out = _group_linear_indices(
                groups, hf_attn.kv_b_proj, prune_out=True
            )
            kv_b_in = _group_linear_indices(
                groups, hf_attn.kv_b_proj, prune_out=False
            )
            o_in = _group_linear_indices(groups, hf_attn.o_proj, prune_out=False)
            if len(o_in):
                raise ValueError(
                    f"Layer {layer}: DeepSeek Q group still contains invalid "
                    f"o_proj.in indices: {o_in.tolist()}."
                )

            new_nope = old_nope - len(nope_local)
            new_rope = old_rope - len(rope_local)
            new_qk = new_nope + new_rope
            removed_complex = set((rope_local // 2).tolist())
            keep_complex = torch.tensor(
                [i for i in range(old_rope // 2) if i not in removed_complex],
                dtype=torch.long,
            )

            print(f"layer={layer} DeepSeek indices from group:")
            print(f"  q_local={q_local.tolist()}")
            print(f"  q_nope={nope_local.tolist()}")
            print(f"  q_rope={rope_local.tolist()}")
            print(f"  kv_a.out={kv_a_out.tolist()}")
            print(f"  kv_b.out={kv_b_out.tolist()}")
            print(f"  kv_b.in={kv_b_in.tolist()}")
            print(f"  o_proj.in={o_in.tolist()}")

            def setup_deepseek(
                attn=attn,
                hf_attn=hf_attn,
                new_nope=new_nope,
                new_rope=new_rope,
                new_qk=new_qk,
                keep_complex=keep_complex,
            ):
                hf_attn.qk_nope_head_dim = new_nope
                hf_attn.qk_rope_head_dim = new_rope
                hf_attn.qk_head_dim = new_qk
                hf_attn.head_dim = new_qk
                hf_attn.scaling = new_qk ** -0.5
                if getattr(attn, "_mla_params_initialized", False):
                    attn._qk_nope_head_dim = new_nope
                    attn._qk_rope_head_dim = new_rope
                    attn._qk_head_dim = new_qk
                attn.register_forward_pre_hook(
                    _make_deepseek_complex_rope_resize_pre_hook(keep_complex),
                    with_kwargs=True,
                )

            structural_setups.append(setup_deepseek)
            structural_shape_modules[f"blocks.{layer}.attn"] = (
                hf_attn.q_proj,
                hf_attn.kv_a_proj_with_mqa,
                hf_attn.kv_b_proj,
                hf_attn.o_proj,
            )
        else:
            old_head_dim = int(hf_attn.head_dim)
            local_idxs = torch.unique(out_idxs.remainder(old_head_dim), sorted=True)
            keep_idxs = torch.tensor(
                [i for i in range(old_head_dim) if i not in set(local_idxs.tolist())],
                dtype=torch.long,
            )

            def setup_standard(
                attn=attn,
                hf_attn=hf_attn,
                keep_idxs=keep_idxs,
            ):
                hf_attn.head_dim = len(keep_idxs)
                attn.register_forward_pre_hook(
                    make_gemma_rope_resize_pre_hook(keep_idxs),
                    with_kwargs=True,
                )

            structural_setups.append(setup_standard)

    print("removed group indices by module:")
    for module_name, removed in removed_idxs_by_module.items():
        print(f"  {module_name}: count={len(removed)}, idxs={removed}")
        
    return local_pruning, groups, structural_setups, structural_shape_modules

def compare_tp_and_transfer_pruning(
    model_bridge,
    prune_task: ModelPruneTask,
    seed: int,
    evaluation_batches: torch.Tensor,
    eval_batch_size: int,
):
    """Compare baseline, transfer masking, and structural Torch-Pruning."""
    local_pruning, groups, structural_setups, structural_shape_modules = prepare_model_for_tp_or_transfer_pruning(
        model_bridge,
        prune_task,
        seed,
        evaluation_batches)

    baseline_metrics = evaluate_language_model(
        model_bridge, evaluation_batches, batch_size=eval_batch_size
    )
    local_pruning.prepare_model_to_transfer_pruning_from_groups(
        groups, rescale=False
    )
    transfer_metrics = evaluate_language_model(
        model_bridge, evaluation_batches, batch_size=eval_batch_size
    )

    model_bridge.reset_hooks()
    for setup in structural_setups:
        setup()
    for group in groups:
        group.prune()

    structural_shapes = {}
    for name, modules in structural_shape_modules.items():
        structural_shapes[name] = {
            "q_proj": tuple(modules[0].weight.shape),
            "kv_a_proj_with_mqa": tuple(modules[1].weight.shape),
            "kv_b_proj": tuple(modules[2].weight.shape),
            "o_proj": tuple(modules[3].weight.shape),
        }
    if structural_shapes:
        print("structural shapes after tp prune:", structural_shapes)

    structural_metrics = evaluate_language_model(
        model_bridge, evaluation_batches, batch_size=eval_batch_size
    )
    comparison = {
        "baseline": baseline_metrics,
        "transfer_activation": transfer_metrics,
        "torch_pruning_structural": structural_metrics,
        "structural_minus_transfer": {
            key: structural_metrics[key] - transfer_metrics[key]
            for key in ("loss", "perplexity")
        },
    }

    del groups, local_pruning
    gc.collect()
    torch.cuda.empty_cache()
    return comparison

#-----------------------------------

def linear_state(module):
    return (module.in_features, module.out_features, tuple(module.weight.shape), id(module.weight))

def debug_group_prune_step_by_step(group):
    '''Prune! And Wrote changed and inconsistent modules after torch_pruning'''
    tracked = {id(dep.target.module): dep.target.module for dep, _ in group if isinstance(dep.target.module, nn.Linear)}
    previous = {key: linear_state(module) for key, module in tracked.items()}
    print("LINEARS BEFORE", previous)
    for step, (dep, idxs) in enumerate(group):
        print(f"\n[{step}] {dep.handler.__name__} target={dep.target.name} idxs={len(idxs)}")
        dep(idxs)
        for key, module in tracked.items():
            current = linear_state(module)
            if current != previous[key]:
                print("CHANGED:", previous[key], "->", current)
                previous[key] = current
            expected = (module.out_features, module.in_features)
            if tuple(module.weight.shape) != expected:
                print("INCONSISTENT AFTER STEP", step, dep.target.name, current)
    return previous

#----------------------------------

class CapturedStages(OrderedDict):
    """Captured tensors plus pruning indices attached to exact hook names."""

    def __init__(self):
        super().__init__()
        self.pruned_indices = {}


def _group_indices_by_attention_stage(attn, groups):
    """Map TP dependencies to hook_in/hook_out of their owning components."""
    targets = {}

    for stage_name, hook in attn.named_modules():
        if not isinstance(hook, HookPoint):
            continue
        if stage_name.endswith(".hook_in"):
            direction = "in"
            component_name = stage_name[:-len(".hook_in")]
        elif stage_name.endswith(".hook_out"):
            direction = "out"
            component_name = stage_name[:-len(".hook_out")]
        else:
            continue

        component = attn.get_submodule(component_name)
        original = getattr(component, "_original_component", component)
        targets[(id(original), direction)] = stage_name

        # Unwrapped parameters appear in TP groups as the Parameter itself,
        # rather than as their owning RMSNorm/module.
        if direction == "out" and isinstance(original, nn.Module):
            for parameter in original.parameters(recurse=False):
                targets[(id(parameter), direction)] = stage_name

    indices_by_stage = defaultdict(set)
    for group in groups:
        DG = group._DG
        for dep, idxs in group:  # type: ignore
            direction = None
            if DG.is_out_channel_pruning_fn(dep.handler):
                direction = "out"
            elif DG.is_in_channel_pruning_fn(dep.handler):
                direction = "in"
            if direction is None:
                continue

            stage_name = targets.get((id(dep.target.module), direction))
            if stage_name is not None:
                indices_by_stage[stage_name].update(map(int, idxs))

    return {
        stage_name: torch.tensor(sorted(indices), dtype=torch.long)
        for stage_name, indices in indices_by_stage.items()
    }


def capture_attention_stages(
    model,
    layer_idx,
    forward_fn,
    groups=None,
    derived_pruned_indices=None,
):
    '''Capture all attention activations in dict'''
    attn = model.blocks[layer_idx].attn
    captured, counts, handles = CapturedStages(), defaultdict(int), []
    if groups is not None:
        captured.pruned_indices.update(
            _group_indices_by_attention_stage(attn, groups)
        )
    if derived_pruned_indices is not None:
        captured.pruned_indices.update({
            name: torch.as_tensor(idxs, dtype=torch.long).unique(sorted=True)
            for name, idxs in derived_pruned_indices.items()
        })

    for relative_name, module in attn.named_modules():
        if not isinstance(module, HookPoint):
            continue

        def capture(value, hook, base_name=relative_name):
            if isinstance(value, torch.Tensor):
                call_idx = counts[base_name]
                counts[base_name] += 1
                name = base_name if call_idx == 0 else f"{base_name}#{call_idx}"
                captured[name] = value.detach().float().cpu().clone()
            return value
        # Added after pruning hooks, so this sees the value consumed downstream.
        module.add_hook(capture, dir="fwd")
        handles.append(module)
    try:
        output = forward_fn(model)
        if isinstance(output, torch.Tensor):
            captured["<model_output>"] = output.detach().float().cpu().clone()
    finally:
        for handle in handles:
            handle.remove_hooks() 
        model.reset_hooks() # dont remove all hooks sometime, so they can be called twice later without handle.remove_hooks()
    return captured

def _align_larger_to_smaller(smaller, larger, raw_idxs, label):
    if smaller.ndim != larger.ndim:
        return None, None
    dims = [d for d, (a, b) in enumerate(zip(smaller.shape, larger.shape)) if a != b]
    if len(dims) != 1:
        return None, None
    dim = dims[0]
    if smaller.shape[dim] >= larger.shape[dim]:
        return None, None
    removed = larger.shape[dim] - smaller.shape[dim]
    pruned = torch.as_tensor(raw_idxs, dtype=torch.long).unique(sorted=True)
    pruned = pruned[(pruned >= 0) & (pruned < larger.shape[dim])]
    if len(pruned) != removed:
        return None, None
    keep = torch.ones(larger.shape[dim], dtype=torch.bool)
    keep[pruned] = False
    return larger.index_select(dim, keep.nonzero(as_tuple=False).flatten()), f"{label}, dim={dim}"

def compare_attention_stages(
    reference,
    candidate,
    atol=1e-4,
    rtol=1e-3,
    record_activations=False,
):
    '''Compare all activation from attn captured dict'''
    first, rows = None, []
    names = list(reference) + [name for name in candidate if name not in reference]
    for stage_idx, name in enumerate(names):
        if name not in reference or name not in candidate:
            print(f"[{stage_idx:02d}] {name:32s} MISSING ref={name in reference} test={name in candidate}")
            first = first or name
            continue
        ref, test, alignment = reference[name], candidate[name], None
        original_shapes = tuple(ref.shape), tuple(test.shape)

        # LinearBridge hook conversion may expose one model as [B, S, H, D]
        # while a structurally-pruned Linear is captured as [B, S, H*D].
        # Canonicalize only this unambiguous one-extra-head-axis case.
        if ref.ndim + 1 == test.ndim and ref.shape[:-1] == test.shape[:-2]:
            test = test.flatten(-2)
            alignment = "flatten candidate [H,D]->[H*D]"
        elif test.ndim + 1 == ref.ndim and test.shape[:-1] == ref.shape[:-2]:
            ref = ref.flatten(-2)
            alignment = "flatten reference [H,D]->[H*D]"

        if ref.shape != test.shape:
            previous_alignment = alignment
            base_name = name.split("#", 1)[0]
            pruned_idxs = getattr(candidate, "pruned_indices", {}).get(base_name)
            index_source = f"candidate.pruned_indices {pruned_idxs}"
            if pruned_idxs is None:
                pruned_idxs = getattr(reference, "pruned_indices", {}).get(base_name)
                index_source = f"reference.pruned_indices {pruned_idxs}"

            if pruned_idxs is not None:
                aligned, index_alignment = _align_larger_to_smaller(
                    ref, test, pruned_idxs, index_source
                )
                if aligned is not None:
                    test = aligned
                    alignment = " + ".join(filter(None, [previous_alignment, index_alignment]))
                else:
                    aligned, index_alignment = _align_larger_to_smaller(
                        test, ref, pruned_idxs, index_source
                    )
                    if aligned is not None:
                        ref = aligned
                        alignment = " + ".join(filter(None, [previous_alignment, index_alignment]))
        if ref.shape != test.shape:
            print(f"[{stage_idx:02d}] {name:32s} SHAPE ref={original_shapes[0]} test={original_shapes[1]}")
            first = first or name
            continue
        delta = test - ref
        close = torch.allclose(ref, test, atol=atol, rtol=rtol)
        max_abs = delta.abs().max().item() if delta.numel() else 0.0
        mean_abs = delta.abs().mean().item() if delta.numel() else 0.0
        rmse = delta.square().mean().sqrt().item() if delta.numel() else 0.0
        rel_l2 = delta.norm().item() / max(ref.norm().item(), 1e-12)
        abs_l2 = delta.square().sum().sqrt().item()
        status = "OK" if close else "DIFF"
        suffix = f" aligned={alignment}" if alignment else ""
        print(f"[{stage_idx:02d}] {name:32s} {status:4s} shape={tuple(ref.shape)} max={max_abs:.3e} mean={mean_abs:.3e} rmse={rmse:.3e} rel_l2={rel_l2:.3e} abs_l2={abs_l2:.3e}{suffix}")
        rows.append({"stage": name, "close": close, "reference_shape": original_shapes[0], "candidate_shape": original_shapes[1], "alignment": alignment, "max_abs": max_abs, "mean_abs": mean_abs, "rmse": rmse, "rel_l2": rel_l2, "abs_l2": abs_l2})
        if (record_activations):
            rows[-1]["reference_activation"] = ref
            rows[-1]["candidate_activation"] = test
        if not close and first is None:
            first = name
    print("\nFIRST DIVERGENCE:", first)
    return first, rows

# ----------------------------------------------------------------------------- #
# Structural-attention helpers. All removed indices are read from the already
# corrected TP group; prune_task is not interpreted a second time here.
# ----------------------------------------------------------------------------- #
def _group_linear_indices(groups, module, prune_out=True):
    result = set()
    for group in groups:
        DG = group._DG
        for dep, idxs in group: # type: ignore
            if dep.target.module is not module:
                continue
            correct_handler = (
                DG.is_out_channel_pruning_fn(dep.handler)
                if prune_out
                else DG.is_in_channel_pruning_fn(dep.handler)
            )
            if correct_handler:
                result.update(map(int, idxs))
    return torch.tensor(sorted(result), dtype=torch.long)


def _make_deepseek_complex_rope_resize_pre_hook(
    keep_complex_idxs: torch.Tensor,
):
    """Resize DeepSeek-V2 complex freqs_cis after structural RoPE pruning."""
    def hook(module, args, kwargs):
        position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is None:
            return args, kwargs
        if not (isinstance(position_embeddings, torch.Tensor)
                and position_embeddings.is_complex()):
            raise TypeError(
                "DeepSeek-V2 structural RoPE hook expected complex freqs_cis, "
                f"got {type(position_embeddings)}."
            )
        kwargs = dict(kwargs)
        kwargs["position_embeddings"] = position_embeddings.index_select(
            -1, keep_complex_idxs.to(position_embeddings.device)
        )
        return args, kwargs
    return hook


def full_attention_test_with_prune(
    bridge: TransformerBridge,
    comparsion_layer: int,
    evaluation_blocks: torch.Tensor,
    prune_task: ModelPruneTask,
    rope_indices=True,
    record_activations=False,
    one_token=True #light test for 1 token
):
    """Compare transfer pruning with structural TP using indices from TP groups.
    NOTE: WORKS ONLY FOR LAYER THAT IN `comparsion_layer`. Please, provide prune_task only for it."""
    
    DEVICE = bridge.device
    evaluation_blocks = evaluation_blocks[:1].to(DEVICE)
    comparison_tokens = evaluation_blocks[0][:1].to(DEVICE)
    if (not one_token):
            comparison_tokens = evaluation_blocks[0].to(DEVICE)

    bridge.eval()
    forward_fn = lambda model: model(comparison_tokens, return_type="logits")

    localPruning = LocalPruning(bridge, evaluation_blocks)
    groups = localPruning.get_torch_pruning_groups(prune_task)

    attn = bridge.blocks[comparsion_layer].attn
    hf_attn = attn._original_component
    is_deepseek_mla = all(
        hasattr(hf_attn, name)
        for name in (
            "q_proj",
            "kv_a_proj_with_mqa",
            "kv_b_proj",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "kv_lora_rank",
        )
    )

    if is_deepseek_mla:
        old_nope_dim = int(hf_attn.qk_nope_head_dim)
        old_rope_dim = int(hf_attn.qk_rope_head_dim)
        old_qk_dim = old_nope_dim + old_rope_dim
        num_heads = int(hf_attn.num_heads)

        q_flat_idxs = _group_linear_indices(groups, hf_attn.q_proj._original_component, prune_out=True)
        kv_a_out_idxs = _group_linear_indices(
            groups, hf_attn.kv_a_proj_with_mqa._original_component, prune_out=True
        )
        kv_b_out_idxs = _group_linear_indices(
            groups, hf_attn.kv_b_proj._original_component, prune_out=True
        )
        kv_b_in_idxs = _group_linear_indices(
            groups, hf_attn.kv_b_proj._original_component, prune_out=False
        )
        o_proj_in_idxs = _group_linear_indices(
            groups, hf_attn.o_proj._original_component, prune_out=False
        )

        if len(q_flat_idxs) == 0:
            raise ValueError(
                f"No q_proj out-channel indices found in groups for layer "
                f"{comparsion_layer}."
            )

        q_local_idxs = torch.unique(q_flat_idxs.remainder(old_qk_dim), sorted=True)
        q_nope_local_idxs = q_local_idxs[q_local_idxs < old_nope_dim]
        q_rope_local_idxs = q_local_idxs[q_local_idxs >= old_nope_dim] - old_nope_dim

        # The group builder has already closed complex pairs. Validate instead
        # of repeating that transformation independently.
        if rope_indices and len(q_rope_local_idxs):
            paired = TorchPruningGroupBuilder.close_complex_rope_pairs(
                q_rope_local_idxs
            )
            if not torch.equal(paired.cpu(), q_rope_local_idxs.cpu()):
                raise ValueError(
                    "DeepSeek Q-RoPE indices in the pruning group are not closed "
                    "over complex pairs."
                )

        removed_nope = len(q_nope_local_idxs)
        removed_rope = len(q_rope_local_idxs)
        if len(q_flat_idxs) != num_heads * (removed_nope + removed_rope):
            raise ValueError(
                "DeepSeek structural pruning requires identical local Q indices "
                "for every head."
            )
        if removed_rope % 2:
            raise ValueError("DeepSeek complex RoPE must remove an even number of real dims.")

        # freqs_cis has one complex coordinate per two real Q/K RoPE dims.
        removed_complex = set((q_rope_local_idxs // 2).tolist())
        keep_complex_idxs = torch.tensor(
            [i for i in range(old_rope_dim // 2) if i not in removed_complex],
            dtype=torch.long,
        )

        derived_pruned_indices = {
            "hook_kv_latent": kv_b_in_idxs,
            "hook_rot_q": q_rope_local_idxs,
            "hook_rot_k": q_rope_local_idxs,
            "hook_q": q_local_idxs,
            "hook_k": q_local_idxs,
        }

        print("DeepSeek indices taken from corrected TP group:")
        for name, values in {
            "q_local_idxs": q_local_idxs,
            "q_flat_idxs": q_flat_idxs,
            "q_nope_local_idxs": q_nope_local_idxs,
            "q_rope_local_idxs": q_rope_local_idxs,
            "kv_a_out_idxs": kv_a_out_idxs,
            "kv_b_out_idxs": kv_b_out_idxs,
            "kv_b_in_idxs": kv_b_in_idxs,
            "o_proj_in_idxs": o_proj_in_idxs,
        }.items():
            print(f"  {name}: count={len(values)}, idxs={values.tolist()}")

        new_nope_dim = old_nope_dim - removed_nope
        new_rope_dim = old_rope_dim - removed_rope
        new_qk_dim = new_nope_dim + new_rope_dim
        if new_nope_dim <= 0 or new_rope_dim <= 0:
            raise ValueError(
                f"Invalid resulting DeepSeek dimensions: nope={new_nope_dim}, "
                f"rope={new_rope_dim}."
            )

        def structural_setup():
            # Update both HF attention and MLAAttentionBridge cached values only
            # after capturing the masked/reference execution.
            hf_attn.qk_nope_head_dim = new_nope_dim
            hf_attn.qk_rope_head_dim = new_rope_dim
            hf_attn.qk_head_dim = new_qk_dim
            hf_attn.head_dim = new_qk_dim
            hf_attn.scaling = new_qk_dim ** -0.5
            if getattr(attn, "_mla_params_initialized", False):
                attn._qk_nope_head_dim = new_nope_dim
                attn._qk_rope_head_dim = new_rope_dim
                attn._qk_head_dim = new_qk_dim
            attn.register_forward_pre_hook(
                _make_deepseek_complex_rope_resize_pre_hook(keep_complex_idxs),
                with_kwargs=True,
            )
    else:
        # Standard Llama/Gemma path, but indices still come from the corrected
        # group instead of being reconstructed from prune_task.
        q_bridge = getattr(attn, "q", None)
        if q_bridge is None:
            q_bridge = getattr(attn, "q_proj", None)
        if q_bridge is None:
            raise AttributeError("Could not locate the attention Q projection bridge.")

        q_module = q_bridge._original_component
        old_head_dim = int(hf_attn.head_dim)
        num_heads = int(hf_attn.config.num_attention_heads)
        q_flat_idxs = _group_linear_indices(groups, q_module, prune_out=True)
        local_idxs = torch.unique(q_flat_idxs.remainder(old_head_dim), sorted=True)

        derived_pruned_indices = {
            "hook_cos": local_idxs,
            "hook_sin": local_idxs,
            "hook_rot_q": local_idxs,
            "hook_rot_k": local_idxs,
        }

        removed = set(local_idxs.tolist())
        keep_idxs = torch.tensor(
            [i for i in range(old_head_dim) if i not in removed],
            dtype=torch.long,
        )

        def structural_setup():
            hf_attn.head_dim = old_head_dim - len(local_idxs)
            attn.register_forward_pre_hook(
                make_gemma_rope_resize_pre_hook(keep_idxs),
                with_kwargs=True,
            )

    # Capture masked execution before changing the physical parameter shapes.
    localPruning.prepare_model_to_transfer_pruning_from_groups(groups, rescale=False)
    instrumentor_stages = capture_attention_stages(
        bridge,
        comparsion_layer,
        forward_fn,
        groups=groups,
        derived_pruned_indices=derived_pruned_indices,
    )

    structural_setup()
    for pruning_group in groups:
        pruning_group.prune()

    tp_stages = capture_attention_stages(
        bridge,
        comparsion_layer,
        forward_fn,
        groups=groups,
        derived_pruned_indices=derived_pruned_indices,
    )
    first_divergence, comparison_rows = compare_attention_stages(
        tp_stages,
        instrumentor_stages,
        record_activations=record_activations,
    )
    return first_divergence, comparison_rows
