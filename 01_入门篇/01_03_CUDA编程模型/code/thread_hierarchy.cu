/**
 * CUDA系列第3篇配套代码：线程层级演示
 * 功能：展示Grid、Block、Thread的层级关系和索引计算
 * 编译：nvcc thread_hierarchy.cu -o thread_hierarchy
 * 运行：./thread_hierarchy
 */

#include <stdio.h>
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

// Kernel 1: 打印线程信息（1D配置）
__global__ void print_thread_info_1d() {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 只打印前几个线程的信息
    if (global_idx < 10) {
        printf("Block %2d, Thread %3d: global_idx = %4d\n",
               blockIdx.x, threadIdx.x, global_idx);
    }
}

// Kernel 2: 打印线程信息（2D配置）
__global__ void print_thread_info_2d(int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int global_idx = y * width + x;
    
    // 只打印部分线程
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x < 4 && threadIdx.y < 4) {
        printf("Block(%d,%d), Thread(%2d,%2d): global_idx = %4d, position = (%d,%d)\n",
               blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, 
               global_idx, x, y);
    }
}

// Kernel 3: 1D向量加法
__global__ void vector_add_1d(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Kernel 4: 2D矩阵加法
__global__ void matrix_add_2d(float* a, float* b, float* c, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        int idx = row * width + col;
        c[idx] = a[idx] + b[idx];
    }
}

// 测试1：1D线程层级
void test_1d_hierarchy() {
    printf("=== 测试1：1D线程层级 ===\n\n");
    
    int threads_per_block = 256;
    int num_blocks = 4;
    
    printf("配置：\n");
    printf("  Grid: [%d]\n", num_blocks);
    printf("  Block: [%d]\n", threads_per_block);
    printf("  总线程数: %d\n\n", num_blocks * threads_per_block);
    
    printf("前10个线程的信息：\n");
    print_thread_info_1d<<<num_blocks, threads_per_block>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("\n");
}

// 测试2：2D线程层级
void test_2d_hierarchy() {
    printf("=== 测试2：2D线程层级 ===\n\n");
    
    dim3 threads_per_block(16, 16);  // 256个线程
    dim3 num_blocks(4, 4);           // 16个Block
    int width = 64;
    
    printf("配置：\n");
    printf("  Grid: [%d, %d]\n", num_blocks.x, num_blocks.y);
    printf("  Block: [%d, %d]\n", threads_per_block.x, threads_per_block.y);
    printf("  总线程数: %d\n\n", 
           num_blocks.x * num_blocks.y * threads_per_block.x * threads_per_block.y);
    
    printf("Block(0,0)的前16个线程信息：\n");
    print_thread_info_2d<<<num_blocks, threads_per_block>>>(width);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("\n");
}

// 测试3：1D向量加法性能对比
void test_1d_vector_add() {
    printf("=== 测试3：不同Block大小的性能对比 ===\n\n");
    
    int n = 10000000;  // 1000万个元素
    size_t bytes = n * sizeof(float);
    
    // 分配主机内存
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    
    // 初始化数据
    for (int i = 0; i < n; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }
    
    // 分配设备内存
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    
    // 测试不同的Block大小
    int block_sizes[] = {64, 128, 256, 512};
    int num_tests = sizeof(block_sizes) / sizeof(int);
    
    printf("数据规模: %d 个元素\n\n", n);
    printf("Block大小  |  Block数量  |  总线程数  |  执行时间(ms)\n");
    printf("--------------------------------------------------------\n");
    
    for (int i = 0; i < num_tests; i++) {
        int threads_per_block = block_sizes[i];
        int num_blocks = (n + threads_per_block - 1) / threads_per_block;
        
        // 预热
        vector_add_1d<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 计时
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        vector_add_1d<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        
        printf("%6d     |  %8d   |  %9d  |  %.3f\n",
               threads_per_block, num_blocks, 
               num_blocks * threads_per_block, milliseconds);
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    // 验证结果
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    bool correct = true;
    for (int i = 0; i < 100; i++) {  // 检查前100个
        if (fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5) {
            correct = false;
            break;
        }
    }
    printf("\n结果验证: %s\n\n", correct ? "✅ 正确" : "❌ 错误");
    
    // 释放内存
    free(h_a); free(h_b); free(h_c);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

// 测试4：2D矩阵加法
void test_2d_matrix_add() {
    printf("=== 测试4：2D矩阵加法 ===\n\n");
    
    int width = 1024, height = 1024;
    size_t bytes = width * height * sizeof(float);
    
    // 分配主机内存
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    
    // 初始化数据
    for (int i = 0; i < width * height; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }
    
    // 分配设备内存
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    
    // 测试不同的Block配置
    printf("矩阵大小: %d × %d\n\n", width, height);
    printf("Block配置  |  Grid配置  |  总线程数  |  执行时间(ms)\n");
    printf("--------------------------------------------------------\n");
    
    struct {
        dim3 block;
        const char* name;
    } configs[] = {
        {{16, 16, 1}, "16×16"},
        {{32, 32, 1}, "32×32"},
        {{32, 16, 1}, "32×16"},
        {{16, 32, 1}, "16×32"}
    };
    
    for (int i = 0; i < 4; i++) {
        dim3 threads_per_block = configs[i].block;
        dim3 num_blocks(
            (width + threads_per_block.x - 1) / threads_per_block.x,
            (height + threads_per_block.y - 1) / threads_per_block.y
        );
        
        // 预热
        matrix_add_2d<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, width, height);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 计时
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        matrix_add_2d<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, width, height);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        
        printf("%8s   |  %3d×%3d   |  %9d  |  %.3f\n",
               configs[i].name, num_blocks.x, num_blocks.y,
               num_blocks.x * num_blocks.y * threads_per_block.x * threads_per_block.y,
               milliseconds);
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    // 验证结果
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    bool correct = true;
    for (int i = 0; i < 100; i++) {
        if (fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5) {
            correct = false;
            break;
        }
    }
    printf("\n结果验证: %s\n\n", correct ? "✅ 正确" : "❌ 错误");
    
    // 释放内存
    free(h_a); free(h_b); free(h_c);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

int main() {
    printf("╔════════════════════════════════════════════════════╗\n");
    printf("║   CUDA线程层级演示程序                             ║\n");
    printf("╚════════════════════════════════════════════════════╝\n\n");
    
    // 检查GPU
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    printf("GPU信息：\n");
    printf("  设备名称: %s\n", prop.name);
    printf("  计算能力: %d.%d\n", prop.major, prop.minor);
    printf("  SM数量: %d\n", prop.multiProcessorCount);
    printf("  每个Block最大线程数: %d\n", prop.maxThreadsPerBlock);
    printf("  最大Grid维度: [%d, %d, %d]\n\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    
    // 运行测试
    test_1d_hierarchy();
    test_2d_hierarchy();
    test_1d_vector_add();
    test_2d_matrix_add();
    
    printf("╔════════════════════════════════════════════════════╗\n");
    printf("║   所有测试完成！                                   ║\n");
    printf("╚════════════════════════════════════════════════════╝\n");
    
    return 0;
}
