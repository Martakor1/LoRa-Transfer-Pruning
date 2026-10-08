from typing import Callable

import torch
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2Attention
from transformer_lens.model_bridge.generalized_components.mla_attention import MLAAttentionBridge
from lora_transfer_pruning.adapter.torch_pruning.index_utils import IndexUtils

class DeepseekV2StructuralSetup:
    
    @staticmethod
    def make_deepseek_complex_rope_resize_pre_hook(
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
    
    @staticmethod
    def get_setup_function_for_deepseek_attn(hf_attn: DeepseekV2Attention, 
                                             attn_module: MLAAttentionBridge, 
                                             bridge_name: str, 
                                             q_idxs_repeated, 
                                             o_in_idxs, 
                                             kv_a_out, 
                                             kv_b_out, 
                                             kv_b_in
                                             ) -> Callable:
        old_nope = int(hf_attn.qk_nope_head_dim)
        old_rope = int(hf_attn.qk_rope_head_dim)
        old_qk = old_nope + old_rope
        num_heads = int(hf_attn.num_heads)
        q_local = torch.unique(q_idxs_repeated.remainder(old_qk), sorted=True)
        nope_local = q_local[q_local < old_nope]
        rope_local = q_local[q_local >= old_nope] - old_nope
        closed_rope = IndexUtils.close_complex_rope_pairs(
            rope_local
        )
        hf_attn._get_name()
        if not torch.equal(closed_rope.cpu(), rope_local.cpu()):
            raise ValueError(
                f"Module {bridge_name}: Q-RoPE group indices are not closed over "
                "DeepSeek complex pairs."
            )
        if len(q_idxs_repeated) != num_heads * len(q_local):
            raise ValueError(
                f"Module {bridge_name}: DeepSeek Q indices are not identical across heads."
            )
        if len(rope_local) % 2:
            raise ValueError(
                f"Module {bridge_name}: odd number of real RoPE dimensions removed."
            )
        if len(o_in_idxs):
            raise ValueError(
                f"Module {bridge_name}: DeepSeek Q group still contains invalid "
                f"o_proj.in indices: {o_in_idxs.tolist()}."
            )
        new_nope = old_nope - len(nope_local)
        new_rope = old_rope - len(rope_local)
        if new_nope <= 0 or new_rope <= 0:
            raise ValueError(
                f"Module {bridge_name}: DeepSeek Q group prunes all Q-NOPE or all Q-RoPE dimensions."
            )
        new_qk = new_nope + new_rope
        removed_complex = set((rope_local // 2).tolist())
        keep_complex = torch.tensor(
            [i for i in range(old_rope // 2) if i not in removed_complex],
            dtype=torch.long,
        )

        print(f"module={bridge_name} DeepSeek indices from group:")
        print(f"  q_local={q_local.tolist()}")
        print(f"  q_nope={nope_local.tolist()}")
        print(f"  q_rope={rope_local.tolist()}")
        print(f"  kv_a.out={kv_a_out.tolist()}")
        print(f"  kv_b.out={kv_b_out.tolist()}")
        print(f"  kv_b.in={kv_b_in.tolist()}")
        print(f"  o_proj.in={o_in_idxs.tolist()}")

        def setup_deepseek(
            attn=attn_module,
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
                DeepseekV2StructuralSetup.make_deepseek_complex_rope_resize_pre_hook(keep_complex),
                with_kwargs=True,
            )

        return setup_deepseek
    
