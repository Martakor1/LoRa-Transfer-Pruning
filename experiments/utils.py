import math
from collections.abc import Callable
from typing import Any

import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

def evaluate_language_model(
    evaluated_bridge: TransformerBridge,
    token_blocks: torch.Tensor,
    batch_size: int = 1,
):
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


def train_model(
    evaluated_bridge: TransformerBridge,
    token_blocks: torch.Tensor,
    num_steps: int,
    *,
    seed = 0,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.0,
    max_grad_norm: float | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    optimizer_factory: Callable[[Any], torch.optim.Optimizer] | None = None,
    print_every: int = 1,
) -> list[dict[str, float | int]]:
    """Fine-tune a TransformerBridge and return per-step metrics.

    AdamW is used by default. Weight decay defaults to zero for controlled
    structural-vs-hooked pruning comparisons, because decoupled weight decay
    updates masked parameters even when their gradients are zero.
    """
    if num_steps < 0:
        raise ValueError(f"num_steps must be non-negative, got {num_steps}.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if len(token_blocks) == 0 and num_steps:
        raise ValueError("token_blocks must not be empty when num_steps > 0.")
    if optimizer is not None and optimizer_factory is not None:
        raise ValueError("Pass either optimizer or optimizer_factory, not both.")

    model_device = next(evaluated_bridge.parameters()).device
    evaluated_bridge.train()

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if optimizer is None:
        if optimizer_factory is not None:
            optimizer = optimizer_factory(evaluated_bridge.parameters())
        else:
            optimizer = torch.optim.AdamW(
                evaluated_bridge.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )

    history: list[dict[str, float | int]] = []
    num_blocks = len(token_blocks)

    for step in range(num_steps):
        indices = (
            torch.arange(
                step * batch_size,
                (step + 1) * batch_size,
                device=token_blocks.device,
            )
            % num_blocks
        )
        batch = token_blocks.index_select(0, indices).to(model_device)

        optimizer.zero_grad(set_to_none=True)
        loss = evaluated_bridge(
            batch,
            return_type="loss",
            use_cache=False,
        )
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise TypeError(
                "TransformerBridge must return a scalar Tensor for "
                f"return_type='loss', got {type(loss)} with "
                f"shape={getattr(loss, 'shape', None)}."
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss at step {step + 1}: {loss.item()}."
            )

        loss.backward()

        if max_grad_norm is not None:
            gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
                evaluated_bridge.parameters(),
                max_grad_norm,
            )
            gradient_norm = float(gradient_norm_tensor.detach().cpu())
        else:
            # squared_norm = torch.zeros((), device="cpu")
            # for parameter in evaluated_bridge.parameters():
            #     if parameter.grad is not None:
            #         squared_norm += parameter.grad.detach().cpu().float().square().sum() #too slow or consume memory on cuda
            gradient_norm = 0 #float(squared_norm.sqrt().cpu())

        optimizer.step()

        loss_value = float(loss.detach().float().cpu())
        perplexity = math.exp(loss_value) if loss_value < 80.0 else math.inf
        current_lr = float(optimizer.param_groups[0]["lr"])
        metrics: dict[str, float | int] = {
            "step": step + 1,
            "loss": loss_value,
            "perplexity": perplexity,
            "gradient_norm": gradient_norm,
            "learning_rate": current_lr,
        }
        history.append(metrics)

        if print_every and (
            (step + 1) % print_every == 0 or step == 0 or step + 1 == num_steps
        ):
            print(
                f"step {step + 1:>4}/{num_steps}: "
                f"loss={loss_value:.6f}, "
                f"perplexity={perplexity:.6f}, "
                f"grad_norm={gradient_norm:.6f}, "
                f"lr={current_lr:.3e}"
            )

    return history
