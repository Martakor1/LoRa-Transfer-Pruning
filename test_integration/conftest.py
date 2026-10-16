import gc

import pytest


def pytest_addoption(parser):
    group = parser.getgroup("pruning integration")
    group.addoption("--run-pruning-integration", action="store_true",
                    help="Load real LLM checkpoints and compare SP/TP on CUDA.")
    group.addoption("--pruning-device", default="cuda:0",
                    help="Single CUDA device for the checkpoint and captures.")
    group.addoption("--pruning-allow-download", action="store_true",
                    help="Allow missing Hugging Face files to be downloaded.")
    group.addoption("--pruning-dtype", choices=("float16", "bfloat16"), default=None,
                    help="Override the FP16 test dtype for numerical diagnostics.")


@pytest.fixture
def pruning_device(request):
    if not request.config.getoption("--run-pruning-integration"):
        pytest.skip("Use --run-pruning-integration to load real LLM checkpoints")
    import torch

    device = torch.device(request.config.getoption("--pruning-device"))
    if device.type != "cuda" or not torch.cuda.is_available():
        pytest.fail("Requested integration run requires a CUDA device")
    with torch.cuda.device(device):
        torch.manual_seed(42)
        try:
            yield device
        finally:
            gc.collect()
            torch.cuda.empty_cache()
