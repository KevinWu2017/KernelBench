import torch
import torch.nn as nn

class Model(nn.Module):
    """
    2D Box Stencil model that applies a box filter to a 2D input tensor.
    
    Box Stencil Operation Overview:
    ==============================
    The box stencil computes the average value within a rectangular neighborhood
    for each point in the input tensor. This is a fundamental operation in:
    - Image processing (smoothing, noise reduction)
    - Numerical simulations (heat diffusion, cellular automata)
    - Computer vision (feature extraction)
    
    Algorithm Steps:
    ================
    1. PADDING PHASE:
       - Input tensor shape: (H, W)
       - Add padding of size R (radius) around all borders
       - Padded tensor shape: (H+2*R, W+2*R)
       - Padding strategy: typically zero-padding or reflection padding
    
    2. STENCIL COMPUTATION PHASE:
       - For each point (i,j) in the original HxW grid:
         a) Extract neighborhood: padded[i:i+2*R+1, j:j+2*R+1]
         b) Apply kernel: element-wise multiply with box filter weights
         c) Reduce: sum all values and divide by neighborhood size
         d) Store result at output[i,j]
    """
    def __init__(self, box_size: int = 5):
        super(Model, self).__init__()
        self.box_size = box_size
        self.radius = box_size // 2
        
        # Create a box filter kernel (uniform weights)
        self.kernel = torch.ones(1, 1, box_size, box_size) / (box_size * box_size)
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # Step 1: Implicit padding and Step 2: Stencil computation
        # torch.stencil handles both padding and neighborhood computation
        return torch.stencil(input_tensor, self.kernel, self.radius)

# Parameters for 2D Box Stencil Operation
# ========================================
H = 1024  # Height of input tensor
W = 1024  # Width of input tensor  
R = 2     # Stencil radius - determines neighborhood size
          # With R=2, each point considers a (2*2+1)x(2*2+1) = 5x5 neighborhood
          # Padding size will also be R=2 pixels on each border
BOX_SIZE = R * 2 + 1  # Total box size: 5x5 neighborhood for averaging

def get_inputs():
    """
    Generate input tensor for the 2D box stencil operation.
    
    Returns:
        list: List containing the input tensor of shape (H, W)
    """
    # Create a random 2D tensor
    input_tensor = torch.randn(H, W)
    return [input_tensor]

def get_init_inputs():
    """
    Get initialization inputs for the model.
    
    Returns:
        list: List containing the box size and box filter tensor
    """
    filter_kernel = torch.ones((BOX_SIZE, BOX_SIZE), dtype=torch.float32) / (BOX_SIZE * BOX_SIZE)
    return [BOX_SIZE, filter_kernel]
