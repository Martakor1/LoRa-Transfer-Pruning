import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

def evaluate_language_model(evaluated_bridge: TransformerBridge, token_blocks, batch_size=1):
    evaluated_bridge.eval()
    model_device = next(evaluated_bridge.parameters()).device
    losses = []

    with torch.inference_mode():
        for start in range(0, len(token_blocks), batch_size):
            batch = token_blocks[start : start + batch_size].to(model_device)
            loss = evaluated_bridge.run_with_hooks(batch, return_type="loss")
            losses.append(loss.detach().float().cpu())

    mean_loss = torch.stack(losses).mean()
    return {
        "loss": mean_loss.item(),
        "perplexity": mean_loss.exp().item(),
    }