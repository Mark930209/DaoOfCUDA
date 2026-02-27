/**
 * CUDA系列第4篇配套代码：内存层级演示
 * 功能：对比不同内存访问模式的性能
 * 编译：nvcc memory_demo.cu -o memory_demo
 * 运行：./memory_demo
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s:%d, %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ========== 测试1：合并访问 vs 非合并访问 ==========

// 合并访问：连续访问
__global__ void coalesced_access(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = data[i] * 2.0f;
    }
}

// 非合并访问：跨步访问
__global__ void uncoalesced_access(float* data, int n, int stride) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    if (i < n) {
        data[i] = data[i] * 2.0f;
    }
}

void test_coalesced_access() {
    printf("=== 测试1：合并访问 vs 非合并访问 ===\n\n");
    
    int n = 10000000;
    size_t bytes = n * sizeof(float);
    
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemset(d_data, 0, bytes));
    
    int threads_per_block = 256;
    
    printf("数据规模: %d 个元素\n\n", n);
    printf("访问模式        |  执行时间(ms)  |  带宽(GB/s)  |  相对性能\n");
    printf("----------------------------------------------------------------\n");
    
    // 测试合并访问
    {
        int num_blocks = (n + threads_per_block - 1) / threads_per_block;
        
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        coalesced_access<<<num_blocks, threads_per_block>>>(d_data, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float bandwidth = (n * sizeof(float) * 2) / (ms / 1000.0) / 1e9;
        
        printf("合并访问(stride=1) |  %.3f         |  %.2f        |  1.00×\n", 
               ms, bandwidth);
        
        float baseline_ms = ms;
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        
        // 测试不同stride的非合并访问
        int strides[] = {2, 4, 8, 16, 32};
        for (int i = 0; i < 5; i++) {
            int stride = strides[i];
            int effective_n = n / stride;
            num_blocks = (effective_n + threads_per_block - 1) / threads_per_block;
            
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));
            
            CUDA_CHECK(cudaEventRecord(start));
            uncoalesced_access<<<num_blocks, threads_per_block>>>(d_data, n, stride);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            bandwidth = (effective_n * sizeof(float) * 2) / (ms / 1000.0) / 1e9;
            
            printf("非合并访问(stride=%2d)|  %.3f         |  %.2f        |  %.2f×\n",
                   stride, ms, bandwidth, baseline_ms / ms);
            
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
        }
    }
    
    CUDA_CHECK(cudaFree(d_data));
    printf("\n");
}

// ========== 测试2：Shared Memory加速 ==========

// Naive矩阵乘法
__global__ void matmul_naive(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 使用Shared Memory的矩阵乘法
#define TILE_SIZE 16

__global__ void matmul_shared(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 加载Tile到Shared Memory
        if (row < N && t * TILE_SIZE + threadIdx.x < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (col < N && t * TILE_SIZE + threadIdx.y < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // 计算
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

void test_shared_memory() {
    printf("=== 测试2：Shared Memory加速矩阵乘法 ===\n\n");
    
    int sizes[] = {128, 256, 512, 1024};
    
    printf("矩阵大小  |  Naive(ms)  |  Shared(ms)  |  加速比\n");
    printf("----------------------------------------------------\n");
    
    for (int i = 0; i < 4; i++) {
        int N = sizes[i];
        size_t bytes = N * N * sizeof(float);
        
        // 分配内存
        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, bytes));
        CUDA_CHECK(cudaMalloc(&d_B, bytes));
        CUDA_CHECK(cudaMalloc(&d_C, bytes));
        
        // 初始化为1（简化）
        CUDA_CHECK(cudaMemset(d_A, 1, bytes));
        CUDA_CHECK(cudaMemset(d_B, 1, bytes));
        
        dim3 threads(16, 16);
        dim3 blocks((N + 15) / 16, (N + 15) / 16);
        
        // 测试Naive版本
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        matmul_naive<<<blocks, threads>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float naive_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&naive_ms, start, stop));
        
        // 测试Shared Memory版本
        CUDA_CHECK(cudaEventRecord(start));
        matmul_shared<<<blocks, threads>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float shared_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&shared_ms, start, stop));
        
        printf("%4d×%4d  |  %8.2f   |  %9.2f   |  %.2f×\n",
               N, N, naive_ms, shared_ms, naive_ms / shared_ms);
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }
    
    printf("\n");
}

// ========== 测试3：Bank Conflict ==========

// 无Bank Conflict
__global__ void no_bank_conflict(float* out) {
    __shared__ float data[256];
    int tid = threadIdx.x;
    
    // 每个Thread访问不同的Bank
    data[tid] = tid;
    __syncthreads();
    
    out[tid] = data[tid];
}

// 有Bank Conflict
__global__ void with_bank_conflict(float* out, int stride) {
    __shared__ float data[256];
    int tid = threadIdx.x;
    
    // 多个Thread访问同一个Bank
    data[tid] = tid;
    __syncthreads();
    
    out[tid] = data[(tid * stride) % 256];
}

void test_bank_conflict() {
    printf("=== 测试3：Bank Conflict影响 ===\n\n");
    
    float *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, 256 * sizeof(float)));
    
    printf("访问模式        |  执行时间(μs)  |  相对性能\n");
    printf("------------------------------------------------\n");
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // 无冲突
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 1000; i++) {
        no_bank_conflict<<<1, 256>>>(d_out);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float no_conflict_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&no_conflict_ms, start, stop));
    float no_conflict_us = no_conflict_ms * 1000.0f / 1000.0f;
    
    printf("无冲突(stride=1) |  %.3f          |  1.00×\n", no_conflict_us);
    
    // 不同stride的冲突
    int strides[] = {2, 4, 8, 16, 32};
    for (int i = 0; i < 5; i++) {
        int stride = strides[i];
        
        CUDA_CHECK(cudaEventRecord(start));
        for (int j = 0; j < 1000; j++) {
            with_bank_conflict<<<1, 256>>>(d_out, stride);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float conflict_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&conflict_ms, start, stop));
        float conflict_us = conflict_ms * 1000.0f / 1000.0f;
        
        printf("有冲突(stride=%2d) |  %.3f          |  %.2f×\n",
               stride, conflict_us, conflict_us / no_conflict_us);
    }
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_out));
    printf("\n");
}

// ========== 测试4：矩阵转置 ==========

// Naive转置
__global__ void transpose_naive(float* in, float* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        out[x * height + y] = in[y * width + x];
    }
}

// 优化转置（使用Shared Memory）
#define TILE_DIM 32

__global__ void transpose_optimized(float* in, float* out, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1避免Bank Conflict
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // 读取到Shared Memory
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    }
    __syncthreads();
    
    // 转置后的坐标
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    // 写入
    if (x < height && y < width) {
        out[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

void test_transpose() {
    printf("=== 测试4：矩阵转置优化 ===\n\n");
    
    int sizes[] = {1024, 2048, 4096};
    
    printf("矩阵大小  |  Naive(ms)  |  Optimized(ms)  |  加速比  |  带宽(GB/s)\n");
    printf("-----------------------------------------------------------------------\n");
    
    for (int i = 0; i < 3; i++) {
        int N = sizes[i];
        size_t bytes = N * N * sizeof(float);
        
        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, bytes));
        CUDA_CHECK(cudaMalloc(&d_out, bytes));
        CUDA_CHECK(cudaMemset(d_in, 1, bytes));
        
        dim3 threads(32, 32);
        dim3 blocks((N + 31) / 32, (N + 31) / 32);
        
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        // Naive版本
        CUDA_CHECK(cudaEventRecord(start));
        transpose_naive<<<blocks, threads>>>(d_in, d_out, N, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float naive_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&naive_ms, start, stop));
        
        // 优化版本
        CUDA_CHECK(cudaEventRecord(start));
        transpose_optimized<<<blocks, threads>>>(d_in, d_out, N, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float opt_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&opt_ms, start, stop));
        
        float bandwidth = (bytes * 2) / (opt_ms / 1000.0) / 1e9;
        
        printf("%4d×%4d  |  %7.2f    |  %11.2f     |  %.2f×   |  %.2f\n",
               N, N, naive_ms, opt_ms, naive_ms / opt_ms, bandwidth);
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
    }
    
    printf("\n");
}

int main() {
    printf("╔════════════════════════════════════════════════════╗\n");
    printf("║   CUDA内存层级演示程序                             ║\n");
    printf("╚════════════════════════════════════════════════════╝\n\n");
    
    // 检查GPU
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    printf("GPU信息：\n");
    printf("  设备名称: %s\n", prop.name);
    printf("  全局内存: %.2f GB\n", prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
    printf("  Shared Memory/Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("  寄存器/Block: %d\n", prop.regsPerBlock);
    printf("  L2 Cache: %.2f MB\n\n", prop.l2CacheSize / 1024.0 / 1024.0);
    
    // 运行测试
    test_coalesced_access();
    test_shared_memory();
    test_bank_conflict();
    test_transpose();
    
    printf("╔════════════════════════════════════════════════════╗\n");
    printf("║   所有测试完成！                                   ║\n");
    printf("╚════════════════════════════════════════════════════╝\n");
    
    return 0;
}
