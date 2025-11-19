import numpy as np

# Parameters for 2D Box Stencil Operation
# ========================================
H = 1024  # Height of input tensor
W = 1024  # Width of input tensor  
R = 3     # Stencil radius - determines neighborhood size
          # With R=3, each point considers a (2*3+1)x(2*3+1) = 7x7 neighborhood
          # Padding size will also be R=3 pixels on each border
BOX_SIZE = R * 2 + 1  # Total box size: 7x7 neighborhood for averaging

def get_init_inputs():
    input_tensor = np.random.rand(H, W).astype(np.float32)
    filter_kernel = np.random.rand(BOX_SIZE, BOX_SIZE).astype(np.float32)

    return (input_tensor, filter_kernel)

def trans_to_tensor(inputs):
    import torch

    input_tensor, filter_kernel = inputs
    input_tensor = torch.tensor(input_tensor, dtype=torch.float32).contiguous()
    filter_kernel = torch.tensor(filter_kernel, dtype=torch.float32).contiguous()

    return (input_tensor, filter_kernel)

def trans_to_numpy_on_cpu(outputs):
    return outputs.cpu().numpy()

def run(inputs, algo_func):
    input_tensor, filter_kernel = inputs
    
    return algo_func(input_tensor, filter_kernel, H, W, R)