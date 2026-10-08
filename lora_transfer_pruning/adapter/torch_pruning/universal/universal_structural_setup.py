import torch


class UniversalStructuralSetup:
    
    @staticmethod
    def make_gemma_rope_resize_pre_hook(keep_idxs: torch.Tensor):

        def hook(module, args, kwargs):
            position_embeddings = kwargs.get("position_embeddings")

            if position_embeddings is None:
                return args, kwargs

            cos, sin = position_embeddings
            idxs = keep_idxs.to(device=cos.device)

            kwargs = dict(kwargs)
            kwargs["position_embeddings"] = (
                cos.index_select(-1, idxs),
                sin.index_select(-1, idxs),
            )

            return args, kwargs

        return hook
    
    @staticmethod
    def get_setup_function_for_ordinary_attn(hf_attn, attn_bridge, q_repeated_idxs):
        old_head_dim = int(hf_attn.head_dim)
        local_idxs = torch.unique(q_repeated_idxs.remainder(old_head_dim), sorted=True)

        removed = set(local_idxs.tolist())
        keep_idxs = torch.tensor(
            [i for i in range(old_head_dim) if i not in removed],
            dtype=torch.long,
        )

        def structural_setup():
            hf_attn.head_dim = old_head_dim - len(local_idxs)
            attn_bridge.register_forward_pre_hook(
                UniversalStructuralSetup.make_gemma_rope_resize_pre_hook(keep_idxs),
                with_kwargs=True,
            )

        return structural_setup