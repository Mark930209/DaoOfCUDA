#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloKernel() {
    printf("Hello, CUDA World! This is thread %d in block %d.\n", 
           threadIdx.x, blockIdx.x);

    printf("Block Dim:(%d,%d,%d).\n", 
           blockDim.x, blockDim.y, blockDim.z);

    printf("Grid Dim:(%d,%d,%d).\n", 
           gridDim.x, gridDim.y, gridDim.z);
}

int main() {
    // Launch the kernel with 2 blocks and 3 threads per block
    helloKernel<<<2, 3>>>();
    
    // Wait for the GPU to finish before accessing on host
    cudaDeviceSynchronize();
    
    return 0;
}