#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s:%d, %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Kernel：向量加法
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    // 计算全局索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 边界检查
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// CPU版本（用于验证）
void vector_add_cpu(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    // 1. 设置问题规模
    int n = 1000000;  // 100万个元素
    size_t bytes = n * sizeof(float);
    
    printf("Vector addition: %d elements\n", n);
    printf("Data size: %.2f MB\n\n", bytes / 1024.0 / 1024.0);
    
    // 2. 分配主机内存
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    float *h_c_ref = (float*)malloc(bytes);  // CPU结果
    
    // 3. 初始化数据
    for (int i = 0; i < n; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }
    
    // 4. 分配设备内存
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    
    // 5. 拷贝数据到GPU
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    
    // 6. 配置kernel启动参数
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    
    printf("Kernel config:\n");
    printf("  Threads per block: %d\n", threads_per_block);
    printf("  Number of blocks: %d\n", num_blocks);
    printf("  Total threads: %d\n\n", num_blocks * threads_per_block);
    
    // 7. 启动kernel并计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    vector_add<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaEventRecord(stop));
    
    // 等待kernel完成
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    printf("GPU execution time: %.3f ms\n", milliseconds);
    
    // 8. 拷贝结果回CPU
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    
    // 9. CPU计算（用于验证）
    vector_add_cpu(h_a, h_b, h_c_ref, n);
    
    // 10. 验证结果
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (fabs(h_c[i] - h_c_ref[i]) > 1e-5) {
            printf("Error: index %d, GPU=%.2f, CPU=%.2f\n", 
                   i, h_c[i], h_c_ref[i]);
            correct = false;
            break;
        }
    }
    
    if (correct) {
        printf("Result: PASSED!\n\n");
    } else {
        printf("Result: FAILED!\n\n");
    }
    
    // 11. 计算性能指标
    float bandwidth = (bytes * 3) / (milliseconds / 1000.0) / 1e9;  // 读A、B，写C
    printf("Effective bandwidth: %.2f GB/s\n", bandwidth);
    
    // 12. 释放内存
    free(h_a); free(h_b); free(h_c); free(h_c_ref);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
}