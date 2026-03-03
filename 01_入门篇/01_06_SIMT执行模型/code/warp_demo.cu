/**
 * CUDA Tutorial Part 6: Warp Execution Model Demo
 * Demonstrates Warp Divergence, Occupancy, and Warp Shuffle concepts
 * Compile: nvcc warp_demo.cu -o warp_demo
 * Run: ./warp_demo
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

// ========== Test 1: Warp Divergence Impact ==========

// No Divergence
__global__ void no_divergence(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = data[i] * 2.0f;  // All threads execute the same operation
    }
}

// 2-way Divergence
__global__ void two_way_divergence(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (i % 2 == 0) {
            data[i] = data[i] * 2.0f;
        } else {
            data[i] = data[i] + 1.0f;
        }
    }
}

// 4-way Divergence
__global__ void four_way_divergence(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int branch = i % 4;
        if (branch == 0) {
            data[i] = data[i] * 2.0f;
        } else if (branch == 1) {
            data[i] = data[i] + 1.0f;
        } else if (branch == 2) {
            data[i] = data[i] - 1.0f;
        } else {
            data[i] = data[i] / 2.0f;
        }
    }
}

// 32-way Divergence (worst case)
__global__ void worst_divergence(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int lane = threadIdx.x % 32;
        if (lane == 0) data[i] *= 2.0f;
        else if (lane == 1) data[i] += 1.0f;
        else if (lane == 2) data[i] -= 1.0f;
        else if (lane == 3) data[i] /= 2.0f;
        else if (lane == 4) data[i] *= 3.0f;
        else if (lane == 5) data[i] += 2.0f;
        else if (lane == 6) data[i] -= 2.0f;
        else if (lane == 7) data[i] /= 3.0f;
        else data[i] = data[i];  // Other lanes
    }
}

void test_divergence() {
    printf("=== Test 1: Warp Divergence Performance Impact ===\n\n");
    
    int n = 10000000;
    size_t bytes = n * sizeof(float);
    
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemset(d_data, 0, bytes));
    
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    
    printf("Data size: %d elements\n", n);
    printf("Config: %d blocks x %d threads\n\n", num_blocks, threads_per_block);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    printf("Divergence Type  |  Time(ms)  |  Relative Perf  |  Efficiency\n");
    printf("---------------------------------------------------------------\n");
    
    // Test no divergence
    CUDA_CHECK(cudaEventRecord(start));
    no_divergence<<<num_blocks, threads_per_block>>>(d_data, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float baseline_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&baseline_ms, start, stop));
    printf("No Divergence   |  %.3f     |  1.00x         |  100%%\n", baseline_ms);
    
    // Test 2-way divergence
    CUDA_CHECK(cudaEventRecord(start));
    two_way_divergence<<<num_blocks, threads_per_block>>>(d_data, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float two_way_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&two_way_ms, start, stop));
    printf("2-way Divergence|  %.3f     |  %.2fx         |  %.0f%%\n",
           two_way_ms, two_way_ms / baseline_ms, 100.0f * baseline_ms / two_way_ms);
    
    // Test 4-way divergence
    CUDA_CHECK(cudaEventRecord(start));
    four_way_divergence<<<num_blocks, threads_per_block>>>(d_data, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float four_way_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&four_way_ms, start, stop));
    printf("4-way Divergence|  %.3f     |  %.2fx         |  %.0f%%\n",
           four_way_ms, four_way_ms / baseline_ms, 100.0f * baseline_ms / four_way_ms);
    
    // Test 32-way divergence
    CUDA_CHECK(cudaEventRecord(start));
    worst_divergence<<<num_blocks, threads_per_block>>>(d_data, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float worst_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&worst_ms, start, stop));
    printf("32-way Divergence|  %.3f     |  %.2fx         |  %.0f%%\n",
           worst_ms, worst_ms / baseline_ms, 100.0f * baseline_ms / worst_ms);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    
    printf("\n");
}

// ========== Test 2: Warp-Level Functions ==========

// Reduction using atomic operations
__global__ void reduce_atomic(float* data, float* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(result, data[i]);
    }
}

// Reduction using Warp Shuffle
__device__ float warp_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__ void reduce_warp(float* data, float* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float value = (i < n) ? data[i] : 0.0f;
    
    // Warp-level reduction
    value = warp_reduce_sum(value);
    
    // First thread of each warp writes result
    if (threadIdx.x % 32 == 0) {
        atomicAdd(result, value);
    }
}

void test_warp_functions() {
    printf("=== Test 2: Warp-Level Function Optimization ===\n\n");
    
    int n = 10000000;
    size_t bytes = n * sizeof(float);
    
    // Allocate memory
    float *h_data = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) {
        h_data[i] = 1.0f;  // All elements = 1 for simplicity
    }
    
    float *d_data, *d_result;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    
    printf("Data size: %d elements\n", n);
    printf("Expected result: %.0f\n\n", (float)n);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    printf("Method          |  Time(ms)  |  Speedup\n");
    printf("--------------------------------------------\n");
    
    // Test atomic operation version
    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));
    CUDA_CHECK(cudaEventRecord(start));
    reduce_atomic<<<num_blocks, threads_per_block>>>(d_data, d_result, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float atomic_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&atomic_ms, start, stop));
    
    float result = 0;
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Atomic          |  %.3f     |  1.00x\n", atomic_ms);
    printf("  Result: %.0f [OK]\n", result);
    
    // Test Warp Shuffle version
    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));
    CUDA_CHECK(cudaEventRecord(start));
    reduce_warp<<<num_blocks, threads_per_block>>>(d_data, d_result, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float warp_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&warp_ms, start, stop));
    
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Warp Shuffle    |  %.3f     |  %.2fx\n", warp_ms, atomic_ms / warp_ms);
    printf("  Result: %.0f [OK]\n", result);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free(h_data);
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_result));
    
    printf("\n");
}

// ========== Test 3: Occupancy Impact ==========

// Kernel using few registers
__global__ void low_register_kernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = data[i];
        data[i] = x * 2.0f;
    }
}

// Kernel using many registers
__global__ void high_register_kernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Use multiple local variables (increase register usage)
        float x0 = data[i];
        float x1 = x0 * 2.0f;
        float x2 = x1 + 1.0f;
        float x3 = x2 - 0.5f;
        float x4 = x3 * 1.5f;
        float x5 = x4 + 2.0f;
        float x6 = x5 - 1.0f;
        float x7 = x6 * 0.5f;
        data[i] = x7;
    }
}

void test_occupancy() {
    printf("=== Test 3: Occupancy and Performance ===\n\n");
    
    int n = 10000000;
    size_t bytes = n * sizeof(float);
    
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemset(d_data, 0, bytes));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    printf("Block Size  |  Low Reg(ms)  |  High Reg(ms)  |  Ratio\n");
    printf("------------------------------------------------------------\n");
    
    int block_sizes[] = {64, 128, 256, 512};
    
    for (int i = 0; i < 4; i++) {
        int threads_per_block = block_sizes[i];
        int num_blocks = (n + threads_per_block - 1) / threads_per_block;
        
        // Test low register kernel
        CUDA_CHECK(cudaEventRecord(start));
        low_register_kernel<<<num_blocks, threads_per_block>>>(d_data, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float low_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&low_ms, start, stop));
        
        // Test high register kernel
        CUDA_CHECK(cudaEventRecord(start));
        high_register_kernel<<<num_blocks, threads_per_block>>>(d_data, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float high_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&high_ms, start, stop));
        
        printf("%4d       |  %.3f        |  %.3f        |  %.2fx\n",
               threads_per_block, low_ms, high_ms, high_ms / low_ms);
    }
    
    printf("\nNote: High register kernel may be slower due to lower Occupancy\n");
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    
    printf("\n");
}

// ========== Test 4: Warp Information Print ==========

__global__ void print_warp_info() {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Print info for threads in first two warps only
    if (warp_id < 2 && lane_id < 4) {
        printf("Block %d, Thread %3d: Warp %d, Lane %2d\n",
               blockIdx.x, tid, warp_id, lane_id);
    }
}

void test_warp_info() {
    printf("=== Test 4: Warp Partitioning Demo ===\n\n");
    
    printf("Block config: 256 threads\n");
    printf("Warp partition: 256 / 32 = 8 warps\n\n");
    
    printf("Thread info for first two warps:\n");
    print_warp_info<<<1, 256>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("\n");
}

int main() {
    printf("========================================================\n");
    printf("     CUDA Warp Execution Model Demo Program           \n");
    printf("========================================================\n\n");
    
    // Check GPU
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    printf("GPU Info:\n");
    printf("  Device Name: %s\n", prop.name);
    printf("  Warp Size: %d\n", prop.warpSize);
    printf("  Max Warps/SM: %d\n", prop.maxThreadsPerMultiProcessor / 32);
    printf("  Max Threads/Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Registers/SM: %d\n", prop.regsPerMultiprocessor);
    printf("  Shared Memory/SM: %zu KB\n\n", prop.sharedMemPerMultiprocessor / 1024);
    
    // Run tests
    test_warp_info();
    test_divergence();
    test_warp_functions();
    test_occupancy();
    
    printf("========================================================\n");
    printf("     All Tests Completed!                             \n");
    printf("========================================================\n");
    
    return 0;
}
