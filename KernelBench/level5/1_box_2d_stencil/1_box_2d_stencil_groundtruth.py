import torch
import torch.nn as nn
import torch.nn.functional as F


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
    
    Example with R=2 (5x5 box):
    ===========================
    Original point (i,j) -> Extract 5x5 neighborhood -> Compute average
    
    Neighborhood pattern:
    [p1][p2][p3][p4][p5]
    [p6][p7][p8][p9][p10]
    [p11][p12][(i,j)][p14][p15]  <- Center point
    [p16][p17][p18][p19][p20]
    [p21][p22][p23][p24][p25]
    
    Result: output[i,j] = (p1 + p2 + ... + p25) / 25
    """
    def __init__(self, box_size: int = 5):
        super(Model, self).__init__()
        self.box_size = box_size
        self.radius = box_size // 2
        
        # Create a box filter kernel (uniform weights)
        self.kernel = torch.ones(1, 1, box_size, box_size) / (box_size * box_size)
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies 2D box stencil operation to the input tensor.
        
        The stencil operation consists of two main steps:
        1. Padding: Add border pixels around the input tensor to handle boundary conditions.
           The padding size equals the stencil radius (R=2), so we add 2 pixels on each side.
           This ensures that every point in the original tensor has a complete neighborhood.
        
        2. Stencil computation: For each point (i,j) in the original tensor:
           - Extract a (2*R+1) x (2*R+1) = 5x5 neighborhood centered at (i,j)
           - Apply the box filter kernel (uniform weights) to compute the average
           - Store the result at position (i,j) in the output tensor
        
        The stencil pattern covers a 5x5 box around each point:
        [*][*][*][*][*]
        [*][*][*][*][*]  
        [*][*][P][*][*]  <- P is the center point
        [*][*][*][*][*]
        [*][*][*][*][*]

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (height, width) for single channel.
                                       Each element represents a data point in the 2D grid.

        Returns:
            torch.Tensor: Output tensor with same shape as input, where each point
                         contains the average of its 5x5 box neighborhood.
        """
        H, W = input_tensor.shape
        output = torch.zeros_like(input_tensor)
        
        # Step 1: PADDING PHASE
        # Add padding of size 'radius' around all borders to handle boundary conditions
        # Padding mode 'reflect' mirrors the border values to avoid edge artifacts
        padded_input = F.pad(input_tensor, (self.radius, self.radius, self.radius, self.radius), mode='reflect')
        
        # Step 2: STENCIL COMPUTATION PHASE
        # Apply box stencil to each point in the original tensor
        for i in range(H):
            for j in range(W):
                # Extract the box neighborhood centered at (i,j)
                # From padded tensor: [i:i+box_size, j:j+box_size] gives us the 5x5 region
                box_region = padded_input[i:i+self.box_size, j:j+self.box_size]
                
                # Compute the average of all values in the 5x5 box neighborhood
                # This is equivalent to applying the uniform box filter kernel
                output[i, j] = torch.mean(box_region)
        
        return output

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
        list: List containing the box size parameter
    """
    return [BOX_SIZE]


# Alternative implementation using manual stencil computation (more explicit)
class ManualBoxStencil(nn.Module):
    """
    Manual implementation of 2D box stencil for educational purposes.
    This version explicitly shows the stencil computation without using conv2d.
    """
    def __init__(self, box_size: int = 3):
        super(ManualBoxStencil, self).__init__()
        self.box_size = box_size
        self.radius = box_size // 2
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Manual 2D box stencil computation.
        
        Args:
            input_tensor (torch.Tensor): Input tensor of shape (H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (H, W)
        """
        H, W = input_tensor.shape
        output = torch.zeros_like(input_tensor)
        
        # Add padding to handle boundaries
        padded_input = F.pad(input_tensor, (self.radius, self.radius, self.radius, self.radius), mode='reflect')
        
        # Apply box stencil to each point
        for i in range(H):
            for j in range(W):
                # Extract box neighborhood
                box_region = padded_input[i:i+self.box_size, j:j+self.box_size]
                # Compute average
                output[i, j] = torch.mean(box_region)
        
        return output