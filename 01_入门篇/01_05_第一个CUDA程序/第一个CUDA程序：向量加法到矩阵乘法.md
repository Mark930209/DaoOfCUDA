# ⚡ 第一个CUDA程序：向量加法到矩阵乘法

> **系列**：CUDA修仙之路  
> **难度**：⭐⭐ (1-5星)  
> **前置知识**：第3篇 - CUDA编程模型，第4篇 - GPU内存体系  
> **预计阅读时间**：18分钟  
> **配套代码**：[GitHub链接](https://github.com/Mark930209/DaoOfCUDA)

---

## 引子：从Hello World到真正的并行计算

在前面的文章中，我们学习了CUDA的理论知识：
- 第1篇：GPU的并行计算能力
- 第2篇：开发环境搭建
- 第3篇：Grid/Block/Thread层级
- 第4篇：内存层级与优化

**现在，是时候动手写代码了！**

本文将带你从零开始编写完整的CUDA程序，包括：
1. 完整的程序结构（主机代码 + 设备代码）
2. 内存管理（分配、拷贝、释放）
3. 错误处理（CUDA API调用检查）
4. 性能测量（计时、带宽计算）
5. 从简单到复杂（向量加法 → 矩阵乘法）

---

**Note:** 在开始正式程序之前，强烈建议大家先回顾一下前文关于Grid、Block、Thread的概念。在文章最后，我们也会着重再讲一下kernel函数中关于GridDim和BlockDim的设置。

## 1. CUDA程序的基本结构

### 1.1 完整的程序流程

![](https://raw.githubusercontent.com/Mark930209/MarkPicRepo/main/imgs/企业微信截图_17724232882754.png)


### 1.2 最小的CUDA程序

```cuda
#include <stdio.h>
#include <cuda_runtime.h>

// Kernel定义：在GPU上执行
__global__ void hello_cuda() {
    printf("Hello from GPU! Block %d, Thread %d\n", 
           blockIdx.x, threadIdx.x);
}

int main() {
    // 启动kernel：1个Block，10个Thread
    hello_cuda<<<1, 10>>>();
    
    // 等待GPU完成
    cudaDeviceSynchronize();
    
    return 0;
}
```

**编译运行**：
```bash
nvcc hello.cu -o hello
./hello
```

**输出**：
```
Hello from GPU! Block 0, Thread 0
Hello from GPU! Block 0, Thread 1
Hello from GPU! Block 0, Thread 2
...
Hello from GPU! Block 0, Thread 9
```

---

## 2. 向量加法：完整实现

### 2.1 问题定义

计算两个向量的和：`C = A + B`

```
A = [1, 2, 3, 4, 5, ...]
B = [10, 20, 30, 40, 50, ...]
C = [11, 22, 33, 44, 55, ...]
```

### 2.2 完整代码

```cuda
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
```

### 2.3 代码详解

#### 错误检查宏

```cuda
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s:%d, %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
```

**为什么需要？**
- CUDA API调用可能失败（内存不足、设备不可用等）
- 不检查错误会导致难以调试的问题

**使用方式**：
```cuda
CUDA_CHECK(cudaMalloc(&d_a, bytes));  // 自动检查错误
```

#### Kernel定义

```cuda
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

**关键点**：
- `__global__`：表示这是一个kernel函数
- `const float*`：输入参数用const（只读）
- `if (i < n)`：边界检查（因为总线程数可能大于n）

#### 计时

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
// kernel执行
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
```

**为什么用cudaEvent而不是CPU计时？**
- kernel启动是异步的，CPU会立即返回
- cudaEvent在GPU上计时，更准确

---

## 3. 矩阵加法：2D索引

### 3.1 问题定义

计算两个矩阵的和：`C = A + B`

```
A = [1  2  3]    B = [10 20 30]    C = [11 22 33]
    [4  5  6]        [40 50 60]        [44 55 66]
    [7  8  9]        [70 80 90]        [77 88 99]
```

### 3.2 Kernel实现

```cuda
__global__ void matrix_add(const float* a, const float* b, float* c, 
                           int width, int height) {
    // 计算2D索引
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 边界检查
    if (col < width && row < height) {
        // 转换为1D索引（行优先）
        int idx = row * width + col;
        c[idx] = a[idx] + b[idx];
    }
}
```

### 3.3 启动配置

```cuda
int width = 1024, height = 1024;

// 2D Block配置
dim3 threads_per_block(16, 16);  // 256个Thread

// 2D Grid配置
dim3 num_blocks(
    (width + threads_per_block.x - 1) / threads_per_block.x,   // X方向
    (height + threads_per_block.y - 1) / threads_per_block.y   // Y方向
);

// 启动kernel
matrix_add<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, width, height);
```

---

## 4. 矩阵乘法：Naive实现

### 4.1 问题定义

计算矩阵乘法：`C = A × B`

```
C[i][j] = Σ A[i][k] * B[k][j]
          k=0 to N-1
```

### 4.2 Naive Kernel

```cuda
__global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
    // 每个Thread计算C的一个元素
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        
        // 点积：A的第row行 × B的第col列
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        
        C[row * N + col] = sum;
    }
}
```

### 4.3 性能分析

**计算复杂度**：
- 每个元素：N次乘法 + N次加法 = 2N FLOP
- 总计算量：N² × 2N = 2N³ FLOP

**内存访问**：
- 每个元素：读取N个A + N个B = 2N次Global Memory访问
- 总访问量：N² × 2N × 4字节 = 8N³ 字节

**问题**：
- Global Memory延迟高（200-400 cycles）
- 大量重复读取（A的每一行被读取N次，B的每一列被读取N次）

**优化方向**：使用Shared Memory缓存（第4篇已讲解）

---

## 5. 错误处理与调试

### 5.1 常见错误类型

| 错误类型 | 原因 | 解决方法 |
|---------|------|---------|
| `cudaErrorInvalidValue` | 参数错误（如NULL指针） | 检查参数 |
| `cudaErrorMemoryAllocation` | 内存不足 | 减小数据规模或释放内存 |
| `cudaErrorInvalidDeviceFunction` | Kernel编译错误 | 检查计算能力匹配 |
| `cudaErrorLaunchOutOfResources` | 资源不足（寄存器/Shared Memory） | 减小Block大小或优化kernel |
| `cudaErrorIllegalAddress` | 非法内存访问 | 检查数组越界 |

### 5.2 Kernel错误检查

**问题**：Kernel启动是异步的，错误不会立即返回。

```cuda
// ❌ 错误：kernel错误不会被检测到
kernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());  // 只检查启动错误

// ✅ 正确：同步后检查执行错误
kernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());        // 检查启动错误
CUDA_CHECK(cudaDeviceSynchronize());   // 等待执行完成，检查执行错误
```

### 5.3 调试技巧

#### 1. 使用printf调试

```cuda
__global__ void debug_kernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 只打印前几个Thread
    if (i < 5) {
        printf("Thread %d: data[%d] = %.2f\n", i, i, data[i]);
    }
}
```

#### 2. 使用cuda-memcheck检测内存错误

```bash
cuda-memcheck ./program
```

**能检测的错误**：
- 数组越界
- 未初始化的内存
- 内存泄漏

#### 3. 使用compute-sanitizer（CUDA 11.8+）

```bash
compute-sanitizer --tool memcheck ./program
```

---

## 6. 性能测量与分析

### 6.1 计时方法对比

| 方法 | 精度 | 适用场景 |
|------|------|---------|
| `cudaEvent` | 微秒级 | GPU kernel计时（推荐） |
| `clock()` | 毫秒级 | CPU代码计时 |
| `std::chrono` | 纳秒级 | CPU代码计时（C++11） |
| `nvprof` | 微秒级 | 完整性能分析 |
| `Nsight Compute` | 纳秒级 | 详细kernel分析 |

### 6.2 带宽计算

```cuda
// 向量加法：读A、B，写C
float bandwidth = (n * sizeof(float) * 3) / (time_s) / 1e9;  // GB/s

// 矩阵乘法：读A、B，写C
float bandwidth = (N * N * sizeof(float) * 3) / (time_s) / 1e9;  // GB/s
```

### 6.3 GFLOPS计算

```cuda
// 矩阵乘法：2N³ FLOP
float gflops = (2.0 * N * N * N) / (time_s) / 1e9;  // GFLOPS
```

### 6.4 性能对比示例

```
向量加法 (N=10M)
├─ CPU (单核):     50 ms    →  240 MB/s
├─ CPU (8核):      10 ms    →  1.2 GB/s
└─ GPU (RTX 4090): 0.5 ms   →  240 GB/s  (200× faster!)

矩阵乘法 (N=1024)
├─ CPU (单核):     5000 ms  →  0.4 GFLOPS
├─ CPU (8核):      800 ms   →  2.7 GFLOPS
├─ GPU (Naive):    50 ms    →  43 GFLOPS   (100× faster!)
└─ GPU (Optimized): 2 ms    →  1075 GFLOPS (2500× faster!)
```

---

## 7. 完整项目结构

### 7.1 推荐的文件组织

```
project/
├── src/
│   ├── main.cu              # 主程序
│   ├── kernels.cu           # Kernel实现
│   └── utils.cu             # 工具函数
├── include/
│   ├── kernels.h            # Kernel声明
│   └── utils.h              # 工具函数声明
├── Makefile                 # 编译脚本
└── README.md                # 说明文档
```

### 7.2 CMakeLists.txt示例

在现代C++和CUDA开发中，**CMake**是更主流的构建工具。以下是一个标准的`CMakeLists.txt`示例：

```cmake
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# 项目名称和支持的语言（必须包含CUDA）
project(CUDA_Project LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置CUDA架构（例如：Ampere架构 sm_80）
set(CMAKE_CUDA_ARCHITECTURES 80)

# 添加可执行文件
add_executable(vector_add 
    src/main.cu 
    src/kernels.cu 
    src/utils.cu
)

# 包含头文件目录
target_include_directories(vector_add PRIVATE include)
```

**编译与运行步骤**：
```bash
mkdir build && cd build
cmake ..
make -j4
./vector_add
```

---

## 8. 最佳实践总结

### 8.1 代码规范

| 规范 | 说明 |
|------|------|
| **错误检查** | 所有CUDA API调用都要检查错误 |
| **边界检查** | Kernel中必须检查数组越界 |
| **const正确性** | 只读参数用const |
| **命名规范** | 主机变量用`h_`前缀，设备变量用`d_`前缀 |
| **注释** | Kernel功能、参数、复杂逻辑都要注释 |

### 8.2 性能优化清单

| 优化项 | 检查方法 |
|--------|---------|
| **合并访问** | 确保Warp内Thread访问连续地址 |
| **Block大小** | 使用256（或32的倍数） |
| **Occupancy** | 使用Nsight Compute查看 |
| **内存带宽** | 计算实际带宽 vs 理论峰值 |
| **计算强度** | FLOPS / 内存访问字节数 |

### 8.3 调试流程

```
1. 编译时错误
   ├─ 语法错误 → 检查CUDA语法
   └─ 链接错误 → 检查库路径

2. 运行时错误
   ├─ CUDA API错误 → 使用CUDA_CHECK
   ├─ Kernel错误 → 使用cuda-memcheck
   └─ 结果错误 → 对比CPU结果

3. 性能问题
   ├─ 使用nvprof分析
   ├─ 使用Nsight Compute详细分析
   └─ 对比理论峰值
```

---

## 总结与思考题

### 核心要点回顾

1. **程序结构**：分配内存 → 拷贝数据 → 启动kernel → 拷贝结果 → 释放内存
2. **错误处理**：所有CUDA API调用都要检查，kernel错误需要同步后检查
3. **性能测量**：使用cudaEvent计时，计算带宽和GFLOPS
4. **从简单到复杂**：向量加法（1D） → 矩阵加法（2D） → 矩阵乘法（计算密集）

### 思考题

1. **为什么kernel启动后CPU会立即返回？** 提示：异步执行
2. **如何判断一个kernel是否正确执行？** 提示：错误检查 + 结果验证
3. **向量加法的性能瓶颈在哪里？** 提示：内存带宽

### 动手练习

1. 实现向量点积（Dot Product）
2. 实现矩阵转置
3. 对比不同Block大小对性能的影响

## 附录: kernel函数中GridDim和BlockDim的设置

在CUDA编程中，启动kernel函数时必须指定执行配置（Execution Configuration），即`<<<GridDim, BlockDim>>>`。这两个参数决定了GPU将启动多少个线程，以及这些线程是如何组织的。

### 1. 核心概念

*   **BlockDim (Block维度)**：定义了一个Block中包含多少个Thread。
    *   最大限制：一个Block最多只能有1024个Thread。
    *   维度：可以是1D、2D或3D（`dim3`类型）。
*   **GridDim (Grid维度)**：定义了一个Grid中包含多少个Block。
    *   最大限制：X维度最大为 $2^{31}-1$，Y和Z维度最大为65535。
    *   维度：可以是1D、2D或3D（`dim3`类型）。

### 2. 常用设置方法

#### 场景一：一维数据处理（如向量加法）

处理长度为 `N` 的一维数组。

```cpp
int N = 1000000;
// 1. 确定Block大小（通常是128, 256, 512等32的整数倍）
int threads_per_block = 256; 

// 2. 计算需要的Block数量（向上取整）
// 公式：(N + threads_per_block - 1) / threads_per_block
int num_blocks = (N + threads_per_block - 1) / threads_per_block;

// 3. 启动Kernel
vector_add<<<num_blocks, threads_per_block>>>(...);
```
*注意：由于向上取整，总线程数（`num_blocks * threads_per_block`）可能会略大于 `N`，因此在kernel内部必须进行边界检查 `if (i < N)`。*

#### 场景二：二维数据处理（如图像处理、矩阵加法）

处理大小为 `width × height` 的二维数据。

```cpp
int width = 1920;
int height = 1080;

// 1. 定义2D的Block大小（总线程数不超过1024）
dim3 threads_per_block(16, 16); // 16 * 16 = 256 threads

// 2. 计算2D的Grid大小（分别在X和Y方向向上取整）
dim3 num_blocks(
    (width + threads_per_block.x - 1) / threads_per_block.x,
    (height + threads_per_block.y - 1) / threads_per_block.y
);

// 3. 启动Kernel
matrix_add<<<num_blocks, threads_per_block>>>(...);
```

#### 场景三：三维数据处理（如体素数据、3D物理模拟）

处理大小为 `X × Y × Z` 的三维数据。

```cpp
int X = 100, Y = 100, Z = 100;

// 1. 定义3D的Block大小
dim3 threads_per_block(8, 8, 8); // 8 * 8 * 8 = 512 threads

// 2. 计算3D的Grid大小
dim3 num_blocks(
    (X + threads_per_block.x - 1) / threads_per_block.x,
    (Y + threads_per_block.y - 1) / threads_per_block.y,
    (Z + threads_per_block.z - 1) / threads_per_block.z
);

// 3. 启动Kernel
volume_process<<<num_blocks, threads_per_block>>>(...);
```

### 3. 最佳实践建议

1.  **BlockDim的选择**：
    *   必须是32的整数倍（因为Warp大小为32）。
    *   常用的值是 128、256 或 512。
    *   不要设置得太小（如32），否则无法隐藏内存延迟；也不要盲目设置为最大值1024，可能会导致寄存器资源不足，降低Occupancy（占用率）。
2.  **GridDim的计算**：
    *   永远记住使用向上取整公式：`(N + block_size - 1) / block_size`。
    *   确保在Kernel代码中配合使用 `if (idx < N)` 进行边界保护。


---

## 参考资料

[CUDA-PG-13.1] NVIDIA. "CUDA C++ Programming Guide v13.1". https://docs.nvidia.com/cuda/cuda-c-programming-guide/

[CUDA-BP-13.1] NVIDIA. "CUDA C++ Best Practices Guide". https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

[Kirk2022-PMPP] Kirk, D. B., & Hwu, W. W. (2022). "Programming Massively Parallel Processors" (4th ed.). Chapter 3: Multidimensional Grids and Data.

---

## 下期预告

**第6篇：🔍 深入SIMT执行模型：Warp、分支与占用率**

下一篇我们将深入学习：
- Warp的执行机制
- Warp Divergence（分支分化）
- Occupancy（占用率）计算
- 如何优化Warp级性能

准备好理解GPU的执行细节了吗？让我们下期见！👋

---

*本文是《CUDA修仙之路》系列的第5篇，共38篇*  
*最后更新：2026年3月2日*
