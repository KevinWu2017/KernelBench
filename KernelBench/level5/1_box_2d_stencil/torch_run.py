import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Load C++ and CUDA code from cpp_source.cpp and cuda_source.cuh
script_dir = os.path.dirname(os.path.abspath(__file__))
cpp_header_path = os.path.join(script_dir, "cpp_func.h")
cpp_file_path = os.path.join(script_dir, "cpp_source.cpp")
cuda_file_path = os.path.join(script_dir, "cuda_source.cuh")

with open(cpp_header_path, "r", encoding="utf-8") as f:
    cpp_header_str = f.read()

with open(cpp_file_path, "r", encoding="utf-8") as f:
    cpp_source_str = f.read()

with open(cuda_file_path, "r", encoding="utf-8") as f:
    cuda_source_str = f.read()

box_stencil = load_inline(
    name='box_stencil',
    cpp_sources=cpp_header_str+cpp_source_str,
    cuda_sources=cuda_source_str,
    functions=['box_stencil_torch', 'box_stencil_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

def algo(input_tensor: torch.Tensor, filter_kernel: torch.Tensor, H: int, W: int, R: int) -> torch.Tensor:
    """
    Perform a 2D box stencil operation on the input tensor using PyTorch.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (H, W)
        filter_kernel (torch.Tensor): Box filter kernel of shape (BOX_SIZE, BOX_SIZE)
        H (int): Height of the input tensor
        W (int): Width of the input tensor
        R (int): Stencil radius

    Returns:
        torch.Tensor: Output tensor after applying the box stencil operation
    """
    if input_tensor.ndim != 2 or input_tensor.shape != (H, W):
        raise ValueError(f"input_tensor must be 2D with shape ({H}, {W})")
    if filter_kernel.ndim != 2 or filter_kernel.shape != (R * 2 + 1, R * 2 + 1):
        raise ValueError(f"filter_kernel must be 2D with shape ({R * 2 + 1}, {R * 2 + 1})")

    return box_stencil.box_stencil_torch(input_tensor, filter_kernel, H, W, R)
