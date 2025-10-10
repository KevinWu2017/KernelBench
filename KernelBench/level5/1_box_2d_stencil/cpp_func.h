#include <cuda_runtime.h>
#include <cooperative_groups.h>

void box_stencil_cuda(float* input, float* filter_kernel, float* output, int H, int W, int radius);
