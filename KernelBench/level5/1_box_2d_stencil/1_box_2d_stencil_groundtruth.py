import torch
import torch.nn as nn


class Model(nn.Module):
    """2D Box stencil implemented with a configurable convolution kernel."""

    def __init__(self, box_size: int = 5, box_filter: torch.Tensor | None = None):
        super().__init__()

        if box_filter is None:
            if box_size <= 0:
                raise ValueError("box_size must be a positive integer when box_filter is not provided")
            kernel = torch.ones(box_size, box_size, dtype=torch.float32) / (box_size * box_size)
        else:
            if box_filter.ndim != 2:
                raise ValueError("box_filter must be a 2D tensor")
            kernel = box_filter.detach().clone().to(torch.float32)

        if kernel.shape[0] != kernel.shape[1]:
            raise ValueError("box_filter must be square for a centered stencil computation")
        if kernel.shape[0] % 2 == 0:
            raise ValueError("box_filter size must be odd to define a stencil center")

        self.box_size = int(kernel.shape[0])
        self.radius = self.box_size // 2

        self.register_buffer("box_filter", kernel)

        padding = (self.radius, self.radius)
        self.conv = nn.Conv2d(1, 1, kernel_size=self.box_filter.shape, bias=False, padding=padding)
        with torch.no_grad():
            self.conv.weight.copy_(self.box_filter.unsqueeze(0).unsqueeze(0))
        self.conv.weight.requires_grad_(False)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply the configured box filter using a convolution module."""

        if input_tensor.ndim != 2:
            raise ValueError("Expected a 2D input tensor of shape (H, W)")

        input_batch = input_tensor.unsqueeze(0).unsqueeze(0)
        output = self.conv(input_batch)
        return output.squeeze(0).squeeze(0)

# Parameters for 2D Box Stencil Operation
# ========================================
H = 1024  # Height of input tensor
W = 1024  # Width of input tensor  
R = 3     # Stencil radius - determines neighborhood size
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
