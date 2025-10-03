import numpy as np

# Parameters for 2D Box Stencil Operation
# ========================================
H = 1024  # Height of input tensor
W = 1024  # Width of input tensor  
R = 2     # Stencil radius - determines neighborhood size
          # With R=2, each point considers a (2*2+1)x(2*2+1) = 5x5 neighborhood
          # Padding size will also be R=2 pixels on each border
BOX_SIZE = R * 2 + 1  # Total box size: 5x5 neighborhood for averaging

def get_init_inputs():
    input_tensor = np.random.rand(H, W)
    filter_kernel = np.random.rand(BOX_SIZE, BOX_SIZE)

    return (input_tensor, filter_kernel)

def run(inputs, algo_func):
    input_tensor, filter_kernel = inputs
    
    return algo_func(input_tensor, filter_kernel, H, W, R)