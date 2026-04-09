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

constexpr int TILE_SIZE = 32;

// 版本一：朴素实现。读取合并，但写入非合并（相邻线程写入地址间隔 height）。
__global__ void transpose_naive(const float* in, float* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        out[x * height + y] = in[y * width + x];
    }
}

// 版本二：引入 Shared Memory 做块内转置，全局写入变为合并访问。
// 但列访问 tile[threadIdx.x][threadIdx.y] 步长恰好 32，产生 Bank Conflict。
__global__ void transpose_shared(const float* in, float* out, int width, int height) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    }
    __syncthreads();

    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    if (x < height && y < width) {
        out[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// 版本三：tile[32][33]（+1 padding），列访问步长变为 33，Bank Conflict 消失。
__global__ void transpose_optimized(const float* in, float* out, int width, int height) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    }
    __syncthreads();

    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    if (x < height && y < width) {
        out[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

static float benchmark_kernel(void (*launch)(const float*, float*, int, int),
                              const float* d_in, float* d_out,
                              int width, int height,
                              int warmup_iters, int benchmark_iters) {
    (void)launch;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

    for (int i = 0; i < warmup_iters; ++i) {
        transpose_naive<<<grid, block>>>(d_in, d_out, width, height);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < benchmark_iters; ++i) {
        launch(d_in, d_out, width, height);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / static_cast<float>(benchmark_iters);
}

static void launch_naive(const float* d_in, float* d_out, int width, int height) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);
    transpose_naive<<<grid, block>>>(d_in, d_out, width, height);
}

static void launch_shared(const float* d_in, float* d_out, int width, int height) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);
    transpose_shared<<<grid, block>>>(d_in, d_out, width, height);
}

static void launch_optimized(const float* d_in, float* d_out, int width, int height) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);
    transpose_optimized<<<grid, block>>>(d_in, d_out, width, height);
}

static bool verify_transpose(const std::vector<float>& in,
                             const std::vector<float>& out,
                             int width, int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float expected = in[y * width + x];
            float actual = out[x * height + y];
            if (std::fabs(expected - actual) > 1e-5f) {
                std::cerr << "Mismatch at (" << x << ", " << y << ")"
                          << ": expected=" << expected
                          << ", actual=" << actual << std::endl;
                return false;
            }
        }
    }
    return true;
}

static double effective_bandwidth_gbps(int width, int height, float avg_ms) {
    double bytes = static_cast<double>(width) * static_cast<double>(height) * sizeof(float) * 2.0;
    double seconds = static_cast<double>(avg_ms) / 1000.0;
    return bytes / seconds / 1.0e9;
}

int main() {
    const int width = 4096;
    const int height = 4096;
    const int warmup_iters = 5;
    const int benchmark_iters = 20;
    const size_t bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(float);

    std::vector<float> h_in(width * height);
    std::vector<float> h_out(width * height, 0.0f);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            h_in[y * width + x] = static_cast<float>(y * width + x);
        }
    }

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    float naive_ms = benchmark_kernel(launch_naive, d_in, d_out, width, height, warmup_iters, benchmark_iters);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    if (!verify_transpose(h_in, h_out, width, height)) {
        return EXIT_FAILURE;
    }

    float shared_ms = benchmark_kernel(launch_shared, d_in, d_out, width, height, warmup_iters, benchmark_iters);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    if (!verify_transpose(h_in, h_out, width, height)) {
        return EXIT_FAILURE;
    }

    float optimized_ms = benchmark_kernel(launch_optimized, d_in, d_out, width, height, warmup_iters, benchmark_iters);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    if (!verify_transpose(h_in, h_out, width, height)) {
        return EXIT_FAILURE;
    }

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Matrix size: " << width << " x " << height << std::endl;
    std::cout << "naive      : " << naive_ms << " ms, "
              << effective_bandwidth_gbps(width, height, naive_ms) << " GB/s" << std::endl;
    std::cout << "shared     : " << shared_ms << " ms, "
              << effective_bandwidth_gbps(width, height, shared_ms) << " GB/s" << std::endl;
    std::cout << "optimized  : " << optimized_ms << " ms, "
              << effective_bandwidth_gbps(width, height, optimized_ms) << " GB/s" << std::endl;

    std::cout << "shared speedup over naive    : " << (naive_ms / shared_ms) << "x" << std::endl;
    std::cout << "optimized speedup over naive : " << (naive_ms / optimized_ms) << "x" << std::endl;

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}