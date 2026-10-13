"""CPU regression models: real HF attention and TL bridge, no checkpoint download."""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
from transformer_lens.model_bridge import TransformerBridge

from lora_transfer_pruning.core.prune_task_type import GroupPruneTask, ModelPruneTask
from lora_transfer_pruning.usecase.local_pruning import LocalPruning
from lora_transfer_pruning.adapter.torch_pruning.universal.gemma4_group_builder import Gemma4GroupBuilder


def tiny_bridge(kind):
    if kind == 'llama':
        config = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=32)
        config._attn_implementation = 'eager'
        model = LlamaForCausalLM(config)
        name = 'meta-llama/Llama-3.1-8B-Instruct'
    else:
        config = Gemma4TextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                                 num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                                 global_head_dim=16, vocab_size=32, vocab_size_per_layer_input=32,
                                 hidden_size_per_layer_input=0, num_kv_shared_layers=2,
                                 layer_types=['sliding_attention', 'full_attention'] * 2)
        config._attn_implementation = 'eager'
        model = Gemma4ForCausalLM(config)
        name = 'google/gemma-4-E4B-it'
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({'<unk>': 0, '<s>': 1, '</s>': 2}, unk_token='<unk>')),
        unk_token='<unk>', bos_token='<s>', eos_token='</s>', pad_token='<unk>',
        name_or_path='tiny-local-tokenizer', add_bos_token=True)
    return TransformerBridge.boot_transformers(name, hf_model=model, tokenizer=tokenizer, dtype=torch.float32).eval()


@pytest.mark.parametrize('kind,layer', [('llama', 0), ('gemma', 0), ('gemma', 1), ('gemma', 2), ('gemma', 3)])
@pytest.mark.parametrize('rows', [[1], .25])
def test_head_pruning_keeps_all_linear_metadata_consistent(kind, layer, rows):
    bridge = tiny_bridge(kind)
    tokens = torch.tensor([[1, 2, 3, 4]])
    local = LocalPruning(bridge, tokens)
    plan = local.get_torch_pruning_groups_and_structural_setups(ModelPruneTask(
        {f'blocks.{layer}.attn.q': GroupPruneTask(None, rows)}, name=None))
    for group in plan.groups:
        for dep, indices in group:
            if isinstance(dep.target.module, torch.nn.Linear):
                assert len(indices) == len(set(indices))
    for setup in plan.structural_setups:
        setup()
    for group in plan.groups:
        group.prune()
    for module in bridge.modules():
        if isinstance(module, torch.nn.Linear):
            assert tuple(module.weight.shape) == (module.out_features, module.in_features)
    output = bridge(input=tokens, use_cache=False, return_type='logits')
    assert output.shape == (1, 4, 32)
    assert torch.isfinite(output).all()
    if kind == 'gemma':
        family = Gemma4GroupBuilder.shared_kv_layers_family(bridge, layer)
        old_dim = 16 if layer in (1, 3) else 8
        removed = 2 if isinstance(rows, list) else int(old_dim // 2 * rows) * 2
        for i in family:
            assert bridge.blocks[i].attn._original_component.head_dim == old_dim - removed
            assert bridge.blocks[i].attn.q_norm._original_component.weight.numel() == old_dim - removed


@pytest.mark.parametrize('rows, error', [([1], None), (1.0, ValueError)])
def test_gemma_group_restores_requires_grad_on_return_and_error(rows, error):
    bridge = tiny_bridge('gemma')
    local = LocalPruning(bridge, torch.tensor([[1, 2, 3]]))
    # Freeze after tracing: changing flags must not be a side effect of group creation.
    for i, parameter in enumerate(bridge.parameters()):
        parameter.requires_grad_(i % 2 == 0)
    expected = {p: p.requires_grad for p in bridge.parameters()}
    task = ModelPruneTask({'blocks.3.attn.q': GroupPruneTask(None, rows)}, name=None)
    if error is None:
        local.get_torch_pruning_groups_and_structural_setups(task)
    else:
        with pytest.raises(error):
            local.get_torch_pruning_groups_and_structural_setups(task)
    assert {p: p.requires_grad for p in bridge.parameters()} == expected


def test_gemma_shared_family_deduplicates_and_rejects_conflicting_masks():
    bridge = tiny_bridge('gemma')
    local = LocalPruning(bridge, torch.tensor([[1, 2, 3]]))
    task = ModelPruneTask({f'blocks.{i}.attn.q': GroupPruneTask(None, [1]) for i in (1, 3)}, name=None)
    plan = local.get_torch_pruning_groups_and_structural_setups(task)
    assert len(plan.groups) == len(plan.structural_setups) == 1
    task.data['blocks.3.attn.q'].rows = [2]
    with pytest.raises(ValueError, match='Conflicting pruning masks'):
        local.get_torch_pruning_groups_and_structural_setups(task)
