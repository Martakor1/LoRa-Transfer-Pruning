from typing import Any, Callable

from lora_transfer_pruning.core.prune_task_type import ModelPruneTask
import torch
from experiments.compare_tp_and_our.compare_utils import prepare_model_for_tp_or_transfer_pruning
from experiments.utils import train_model
from experiments.compare_tp_and_our.compare_utils import create_prune_task
from transformer_lens.model_bridge.bridge import TransformerBridge


def prepare_model_to_transfer_pruning(local_pruning, groups, rescale):
    local_pruning.prepare_model_to_transfer_pruning_from_groups(
        groups, rescale=rescale
    )
    
def prepare_model_to_torch_pruning(model_bridge, groups, structural_setups, structural_shape_modules): 
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
        
def get_learn_stat_for_some_pruning(
    model_bridge,
    prune_task: ModelPruneTask,
    seed: int,
    evaluation_batches: torch.Tensor,
    rescale: bool = False,
    is_torch_pruning: bool = False,
    num_steps: int = 10,
    batch_size: int = 4,
    optimizer_factory: Callable[[Any], torch.optim.Optimizer] | None = None,
    print_every: int = 1
    ):
    local_pruning, groups, structural_setups, structural_shape_modules = prepare_model_for_tp_or_transfer_pruning(
        model_bridge,
        prune_task,
        seed,
        evaluation_batches
    )
    
    if (is_torch_pruning):
        prepare_model_to_torch_pruning(model_bridge, groups, structural_setups, structural_shape_modules)
    else:
        prepare_model_to_transfer_pruning(local_pruning, groups, rescale)
    
    learning_history = train_model(
        model_bridge, evaluation_batches, num_steps, batch_size=batch_size, optimizer_factory=optimizer_factory,
        print_every=print_every
    )
    return learning_history

def full_load_prune_learn_pipeline(
    model_name: str,
    model_load_func: Callable[..., tuple[torch.nn.Module, Any]],
    evaluation_batches: torch.Tensor,
    fraction_attn_layers: list[int],
    fraction_mlp_layers: list[int],
    attn_out_fraction: list[int] | float,
    mlp_out_fraction: list[int] | float,
    q_proj_name: str = "q_proj",
    mlp_up_proj_name: str = "up_proj",
    is_torch_pruning: bool = False,
    rescale: bool = False,
    seed: int = 0,
    batch_size: int = 4,
    optimizer_factory: Callable[[Any], torch.optim.Optimizer] | None = None,
    num_steps: int = 10,
    print_every: int = 1
    ):
    
    model = model_load_func()
    bridge = TransformerBridge.boot_transformers(
        model_name,
        hf_model=model,
        dtype=torch.float16,
    )
    
    prune_task = create_prune_task(fraction_attn_layers, fraction_mlp_layers, attn_out_fraction, mlp_out_fraction, q_proj_name, mlp_up_proj_name)
    
    learning_history = get_learn_stat_for_some_pruning(bridge, 
                                                       prune_task, 
                                                       seed, 
                                                       evaluation_batches, 
                                                       rescale=rescale, 
                                                       is_torch_pruning=is_torch_pruning,
                                                       batch_size=batch_size,
                                                       optimizer_factory=optimizer_factory,
                                                       num_steps=num_steps,
                                                       print_every=print_every)
    return learning_history