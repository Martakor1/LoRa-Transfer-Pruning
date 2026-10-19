"""Reproduce the local Torch-Pruning 1.6.1 concat-dim workaround in CPU CI.

Tracked in Martakor1/LoRa-Transfer-Pruning#16 and VainF/Torch-Pruning#532.
Remove this patch once the project adopts and tests an upstream fix.
"""
from importlib.metadata import version
from pathlib import Path

import torch_pruning.dependency.constants as constants

if version("torch-pruning") != "1.6.1":
    raise RuntimeError("Recheck the concat-dim workaround when upgrading Torch-Pruning")

path = Path(constants.__file__)
source = path.read_text()
old = "MAX_VALID_DIM = 100"
new = "MAX_VALID_DIM = 1 << 64"
if old in source:
    path.write_text(source.replace(old, new, 1))
elif new not in source:
    raise RuntimeError("Unexpected Torch-Pruning constants; review the concat-dim patch")
