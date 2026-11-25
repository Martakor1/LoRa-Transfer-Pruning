from types import SimpleNamespace
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwen_torchao_multiadapter import SharedFrozenPrefix, LAYERS


def test_prefix_reuse_is_pair_local_and_restorable():
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = SimpleNamespace(_original_component=SimpleNamespace(attention_dropout=0))
            self.calls = 0
        def forward(self, x):
            self.calls += 1
            return x + 1

    blocks = [Block() for _ in range(min(LAYERS))]
    cache = SharedFrozenPrefix(SimpleNamespace(blocks=blocks,
        original_model=SimpleNamespace(config=SimpleNamespace(use_cache=False, output_router_logits=False))))
    def run():
        x = torch.zeros(2, 3)
        for block in blocks:
            x = block(x)
        return x

    reference = run()
    cache.enabled = True
    first, second = run(), run()
    assert torch.equal(first, reference) and torch.equal(second, reference)
    assert cache.reused_calls == min(LAYERS)
    assert all(block.calls == 2 for block in blocks)
    cache.enabled, cache.cached = False, None
    assert torch.equal(run(), reference)
    cache.restore()
    assert torch.equal(run(), reference)
    assert all(block.calls == 4 for block in blocks)
