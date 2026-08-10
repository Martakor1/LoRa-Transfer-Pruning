import torch


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

rope_resize_handles = []