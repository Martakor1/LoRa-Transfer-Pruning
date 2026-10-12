import sys
import torch
import torch.nn.functional as F

#code for debug with Nsight to see kernels and operations
#nsys profile   --trace=cuda,nvtx,cublas   --output=/tmp/o_proj_trailing   python gemm_probe.py trailing


case = sys.argv[1]
device = torch.device("cuda:3")

data = torch.load(
    "/tmp/o_proj_gemm_inputs.pt",
    map_location="cpu",
)

mapping = {
    "interspersed": (
        "x_interspersed",
        "w_interspersed",
    ),
    "trailing": (
        "x_trailing",
        "w_trailing",
    ),
    "compact": (
        "x_compact",
        "w_compact",
    ),
}

x_name, w_name = mapping[case]

x = data[x_name].to(device).contiguous()
w = data[w_name].to(device).contiguous()

bias = data["bias"]
if bias is not None:
    bias = bias.to(device).contiguous()

# Warm-up outside the profiled range.
for _ in range(20):
    F.linear(x, w, bias)

torch.cuda.synchronize()

# Nsight Compute can be started with --profile-from-start off.
torch.cuda.cudart().cudaProfilerStart()

for _ in range(5):
    output = F.linear(x, w, bias)

torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()

print(case, output.shape, output.flatten()[:4])