#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include "cuda_source.cuh"

int main(int argc, char** argv) {
    // 解析三个参数 H, W, R
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <H> <W> <R>" << std::endl;
        return 1;
    }
    int H = std::stoi(argv[1]);
    int W = std::stoi(argv[2]);
    int R = std::stoi(argv[3]);

    // 初始化input((H+2R)*(W+2R))和filter_kernel((2R+1)*(2R+1))，均为随机值
    // std::vector<float> input((H + 2 * R) * (W + 2 * R));
    // std::vector<float> filter_kernel((2 * R + 1) * (2 * R + 1));
    std::vector<float> input((H + 2 * R) * (W + 2 * R), 1.0f); // 示例输入
    std::vector<float> filter_kernel((2 * R + 1) * (2 * R + 1), 1);
    // 随机初始化 input 和 filter_kernel
    for (auto& v : input) v = static_cast<float>(rand()) / RAND_MAX;
    for (auto& v : filter_kernel) v = static_cast<float>(rand()) / RAND_MAX;
    std::vector<float> output(H * W, 0.0f);

    // gpu上分配内存并拷贝数据
    float *d_input, *d_filter_kernel, *d_output;
    cudaMalloc(&d_input, input.size() * sizeof(float));
    cudaMalloc(&d_filter_kernel, filter_kernel.size() * sizeof(float));
    cudaMalloc(&d_output, output.size() * sizeof(float));
    cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter_kernel, filter_kernel.data(), filter_kernel.size() * sizeof(float), cudaMemcpyHostToDevice);

    // 调用 CUDA 核函数
    box_stencil_cuda(d_input, d_filter_kernel, d_output, H, W, R);

    cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_filter_kernel);
    cudaFree(d_output);

    return 0;
}