/**
 * CUDA系列第1篇配套代码：向量加法
 * 功能：对比CPU和GPU实现向量加法的性能差异
 * 编译：nvcc vector_add.cu -o vector_add
 * 运行：./vector_add
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

// 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s:%d, %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// CPU版本：向量加法
void vector_add_cpu(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// CUDA Kernel：向量加法
__global__ void vector_add_cuda(float* a, float* b, float* c, int n) {
    // 计算当前线程的全局索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 边界检查
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// 验证结果是否正确
bool verify_result(float* cpu_result, float* gpu_result, int n) {
    for (int i = 0; i < n; i++) {
        if (fabs(cpu_result[i] - gpu_result[i]) > 1e-5) {
            printf("❌ 结果不匹配！索引 %d: CPU=%.2f, GPU=%.2f\n", 
                   i, cpu_result[i], gpu_result[i]);
            return false;
        }
    }
    return true;
}

// 计时函数（毫秒）
double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

int main() {
    printf("=== CUDA向量加法性能对比 ===\n\n");
    
    // 1. 设置问题规模
    int n = 10000000;  // 1000万个元素
    size_t bytes = n * sizeof(float);
    
    printf("📊 问题规模：%d 个元素 (%.2f MB)\n\n", n, bytes / 1024.0 / 1024.0);
    
    // 2. 分配主机内存
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c_cpu = (float*)malloc(bytes);  // CPU结果
    float *h_c_gpu = (float*)malloc(bytes);  // GPU结果
    
    // 3. 初始化数据
    printf("🔧 初始化数据...\n");
    for (int i = 0; i < n; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }
    
    // ========== CPU计算 ==========
    printf("\n⏱️  CPU计算中...\n");
    double cpu_start = get_time_ms();
    vector_add_cpu(h_a, h_b, h_c_cpu, n);
    double cpu_time = get_time_ms() - cpu_start;
    printf("✅ CPU时间: %.2f ms\n", cpu_time);
    
    // ========== GPU计算 ==========
    printf("\n⏱️  GPU计算中...\n");
    
    // 4. 分配设备内存
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    
    // 5. 拷贝数据到GPU
    double gpu_start = get_time_ms();
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    double copy_to_gpu_time = get_time_ms() - gpu_start;
    
    // 6. 配置kernel启动参数
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    printf("   Grid配置: %d blocks × %d threads = %d threads\n", 
           blocks, threads_per_block, blocks * threads_per_block);
    
    // 7. 启动kernel
    double kernel_start = get_time_ms();
    vector_add_cuda<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaDeviceSynchronize());  // 等待kernel完成
    double kernel_time = get_time_ms() - kernel_start;
    
    // 8. 拷贝结果回CPU
    double copy_back_start = get_time_ms();
    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost));
    double copy_back_time = get_time_ms() - copy_back_start;
    
    double gpu_total_time = copy_to_gpu_time + kernel_time + copy_back_time;
    
    printf("✅ GPU总时间: %.2f ms\n", gpu_total_time);
    printf("   ├─ 数据拷贝到GPU: %.2f ms\n", copy_to_gpu_time);
    printf("   ├─ Kernel执行: %.2f ms\n", kernel_time);
    printf("   └─ 结果拷贝回CPU: %.2f ms\n", copy_back_time);
    
    // 9. 验证结果
    printf("\n🔍 验证结果...\n");
    if (verify_result(h_c_cpu, h_c_gpu, n)) {
        printf("✅ 结果正确！CPU和GPU计算结果一致\n");
    } else {
        printf("❌ 结果错误！\n");
    }
    
    // 10. 性能对比
    printf("\n📈 性能对比：\n");
    printf("   CPU时间:        %.2f ms\n", cpu_time);
    printf("   GPU总时间:      %.2f ms (加速比: %.1fx)\n", 
           gpu_total_time, cpu_time / gpu_total_time);
    printf("   GPU纯计算时间:  %.2f ms (加速比: %.1fx)\n", 
           kernel_time, cpu_time / kernel_time);
    
    // 计算吞吐量
    double cpu_throughput = (n * sizeof(float) * 2) / (cpu_time / 1000.0) / 1e9;  // GB/s
    double gpu_throughput = (n * sizeof(float) * 2) / (kernel_time / 1000.0) / 1e9;
    printf("\n💾 内存吞吐量：\n");
    printf("   CPU: %.2f GB/s\n", cpu_throughput);
    printf("   GPU: %.2f GB/s\n", gpu_throughput);
    
    // 11. 释放内存
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    
    printf("\n✨ 程序执行完成！\n");
    
    return 0;
}
