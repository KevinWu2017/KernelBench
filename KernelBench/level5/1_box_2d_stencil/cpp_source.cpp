/*
 * @brief Perform a 2D box stencil operation on the input tensor.
 * 
 * @param input Input tensor of shape (H, W).
 * @param filter_kernel Filter kernel tensor of shape (K, K).
 * @param H Height of the input tensor.
 * @param W Width of the input tensor.
 * @param radius Radius of the stencil operation.
 * @return torch::Tensor Output tensor after applying the box stencil operation.
 */
torch::Tensor box_stencil_torch(torch::Tensor input, torch::Tensor filter_kernel, int H, int W, int radius){
    int K = radius / 2; // Assuming square kernel

    torch::Tensor output = torch::zeros({H, W}, input.options());
    box_stencil_cuda(input.data_ptr<float>(), filter_kernel.data_ptr<float>(), output.data_ptr<float>(), H, W, radius);
    return output;
}