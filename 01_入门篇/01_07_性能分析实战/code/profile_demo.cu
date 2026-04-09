#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)            \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

__global__ void warmup_kernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = data[i] * 1.00001f + 0.0001f;
    }
}

__global__ void saxpy_kernel(float a, const float* x, const float* y, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a * x[i] + y[i];
    }
}

__global__ void tiny_kernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = data[i];
        #pragma unroll 4
        for (int k = 0; k < 4; ++k) {
            v = v * 1.0001f + 0.001f;
        }
        data[i] = v;
    }
}

static float run_kernel_average_ms(int iters, int grid, int block, float a,
                                   const float* d_x, const float* d_y, float* d_out, int n) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        saxpy_kernel<<<grid, block>>>(a, d_x, d_y, d_out, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / static_cast<float>(iters);
}

int main() {
    const int n = 1 << 24;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    const int block = 256;
    const int grid = (n + block - 1) / block;
    const int iterations = 20;

    std::vector<float> h_x(n, 1.0f);
    std::vector<float> h_y(n, 2.0f);
    std::vector<float> h_out(n, 0.0f);

    float *d_x = nullptr, *d_y = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out, h_out.data(), bytes, cudaMemcpyHostToDevice));

    // 预热：避免把 CUDA 上下文初始化等一次性开销算入正式测量。
    warmup_kernel<<<grid, block>>>(d_out, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    float avg_ms = run_kernel_average_ms(iterations, grid, block, 2.0f, d_x, d_y, d_out, n);
    std::cout << "[Baseline] saxpy average kernel time: "
              << std::fixed << std::setprecision(4) << avg_ms << " ms" << std::endl;

    // 阶段 2（反面教材）：大量极小 kernel + 每次同步，制造 GPU 空闲碎片。
    // 用 Nsight Systems 观察 timeline 可以看到明显的调度开销。
    for (int i = 0; i < 200; ++i) {
        tiny_kernel<<<1, 256>>>(d_out, 256);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    double checksum = 0.0;
    for (int i = 0; i < 1024; ++i) {
        checksum += h_out[i];
    }
    std::cout << "Checksum(first 1024 elems): " << checksum << std::endl;

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}