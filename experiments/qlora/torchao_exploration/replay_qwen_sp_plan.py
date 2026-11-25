"""Replay this experiment's saved DG mask manifest for repeated speed probes.

No new masks or dependency heuristics: indices come from the completed original
LocalPruning experiment. Only valid for its unchanged Qwen checkpoint/architecture.
"""
import json
from experiments.qlora.torchao_exploration.train_qwen_torchao_simpruning import (
    TransformerBridge, ArchitectureAdapterFactory, Qwen3MoeWithExpertBridges,
    GPTQ_MODEL, HERE, audit_masks, ModelPruneTask, GroupPruneTask,
)
from lora_transfer_pruning.core.pruning_instrumentor import PruningInstrumentor
from lora_transfer_pruning.core.torch_pruning_group_mapper import TorchPruningGroupMapper
import torch


def wrap_qwen(model):
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    ArchitectureAdapterFactory.register_adapter('Qwen3MoeForCausalLM', Qwen3MoeWithExpertBridges)
    bridge = TransformerBridge.boot_transformers(GPTQ_MODEL, hf_model=model.get_base_model(), dtype=torch.float16)
    assert trainable <= {id(p) for p in model.parameters()}
    for p in model.parameters():
        p.requires_grad_(id(p) in trainable)
    return bridge


def apply_saved_plan(model, bridge, train):
    source = HERE / 'slop_experiments/6_torchao_simpruning_train/torchao_sft_simpruning/pruning_plan.json'
    boundaries = []
    roots = {}
    for row in json.loads(source.read_text()):
        name, direction = row['hook'].rsplit('.', 1)
        module = bridge.get_submodule(name)
        boundaries.append((getattr(module, direction), row['removed_indices']))
        if direction == 'hook_in':
            assert name.endswith(('.attn.o', '.down_proj'))
            continue  # Same mapper rule: already zeroed by upstream output masks.
        assert direction == 'hook_out'
        ids = torch.tensor(row['removed_indices'], device='cuda')
        PruningInstrumentor.prepare_linear_for_pruning(module, None, ids, rescale=False,
            active_adapters_source=lambda: model.active_adapters, adapter_name='default')
        if name.endswith(('.attn.q', '.up_proj')):
            roots[name] = GroupPruneTask(None, 0.3)
    TorchPruningGroupMapper.prepare_all_norms(bridge)
    tokens = torch.tensor(train[0]['input_ids'][:16], device='cuda')[None]
    audit = audit_masks(model, bridge, ModelPruneTask(roots, name='default'), boundaries, tokens)
    return dict(source_plan=str(source), mask_audit=audit)


def prepare_saved_plan(model, tokenizer, train, trainer, output):
    return apply_saved_plan(model, wrap_qwen(model), train)
