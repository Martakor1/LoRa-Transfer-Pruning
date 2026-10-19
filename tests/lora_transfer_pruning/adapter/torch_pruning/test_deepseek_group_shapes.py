"""Real MLA dependency graph regression without a pretrained checkpoint."""
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import DeepseekV2Config, DeepseekV2ForCausalLM, PreTrainedTokenizerFast
from transformer_lens.model_bridge import TransformerBridge

from lora_transfer_pruning.core.prune_task_type import GroupPruneTask, ModelPruneTask
from lora_transfer_pruning.usecase.local_pruning import LocalPruning


@pytest.mark.parametrize('indices', [[1], [18], [1, 18]])
def test_deepseek_q_mask_and_unpruned_latent_boundaries(indices):
    config = DeepseekV2Config(
        hidden_size=64, intermediate_size=128, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=32,
        q_lora_rank=None, kv_lora_rank=32, qk_nope_head_dim=16,
        qk_rope_head_dim=8, v_head_dim=16, n_routed_experts=2, num_experts_per_tok=1, first_k_dense_replace=1,
    )
    config.architectures = ["DeepseekV2ForCausalLM"]
    config._attn_implementation = 'eager'
    model = DeepseekV2ForCausalLM(config).half()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({'<unk>': 0, '<s>': 1, '</s>': 2}, unk_token='<unk>')),
        unk_token='<unk>', bos_token='<s>', eos_token='</s>', pad_token='<unk>',
        name_or_path='tiny-local-tokenizer', add_bos_token=True,
    )
    bridge = TransformerBridge.boot_transformers(
        'deepseek-ai/DeepSeek-V2-Lite-Chat', hf_model=model,
        tokenizer=tokenizer, dtype=torch.float16,
    ).eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    local = LocalPruning(bridge, tokens)
    plan = local.get_torch_pruning_groups_and_structural_setups(ModelPruneTask({
        'blocks.0.attn.q_proj': GroupPruneTask(cols=None, rows=indices),
    }))
    attn = bridge.blocks[0].attn
    q = attn.q_proj.original_component
    nope_removed = int(1 in indices)
    rope_removed = 2 if 18 in indices else 0
    q_local = ([1] if nope_removed else []) + ([18, 19] if rope_removed else [])
    expected_indices = sorted(h * 24 + i for h in range(4) for i in q_local)
    q_ops = [(dep, idxs) for dep, idxs in plan.groups[0] if dep.target.module is q]
    assert len(q_ops) == 1
    assert sorted(q_ops[0][1]) == expected_indices
    local.prepare_model_to_transfer_pruning_from_groups(plan.groups, rescale=False)
    try:
        reference = bridge(input=tokens, use_cache=False, return_type="logits").detach()
    finally:
        bridge.reset_hooks()
    for setup in plan.structural_setups:
        setup()
    for group in plan.groups:
        group.prune()
    assert q.out_features == 4 * (24 - nope_removed - rope_removed)
    assert attn.kv_b_proj.original_component.in_features == 32
    assert attn.kv_b_proj.original_component.out_features == 4 * (32 - nope_removed)
    assert attn.kv_a_proj_with_mqa.original_component.out_features == 40 - rope_removed
    assert attn.o_proj.original_component.in_features == 64
    result = bridge(input=tokens, use_cache=False, return_type='logits')
    assert result.shape == (1, 5, 32)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, reference, atol=1e-3, rtol=1e-3)
