
def algo(input_tensor, filter_kernel, H, W, R):
    """
    Perform a 2D box stencil operation on the input tensor.
    
    Args:
        input_tensor (np.ndarray): Input tensor of shape (H, W)
        filter_kernel (np.ndarray): Box filter kernel of shape (BOX_SIZE, BOX_SIZE)
        H (int): Height of the input tensor
        W (int): Width of the input tensor
        R (int): Stencil radius
    
    Returns:
        np.ndarray: Output tensor after applying the box stencil operation
    """
    import numpy as np
    
    BOX_SIZE = R * 2 + 1  # Total box size: 5x5 neighborhood for averaging
    
    # Step 1: Padding Phase
    padded_tensor = np.pad(input_tensor, pad_width=R, mode='constant', constant_values=0)
    
    # Step 2: Stencil Computation Phase
    output_tensor = np.zeros((H, W), dtype=input_tensor.dtype)
    
    for i in range(H):
        for j in range(W):
            # Extract the neighborhood
            neighborhood = padded_tensor[i:i+BOX_SIZE, j:j+BOX_SIZE]
            # Apply the box filter (element-wise multiplication and sum)
            output_value = np.sum(neighborhood * filter_kernel)
            output_tensor[i, j] = output_value
    
    return output_tensor