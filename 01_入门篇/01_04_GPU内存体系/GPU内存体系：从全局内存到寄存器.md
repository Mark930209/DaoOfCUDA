# 💾 GPU内存体系：从全局内存到寄存器

> **系列**：CUDA修仙之路  
> **难度**：⭐⭐⭐ (1-5星)  
> **前置知识**：第3篇 - CUDA编程模型  
> **预计阅读时间**：20分钟  
> **配套代码**：[GitHub链接](https://github.com/Mark930209/DaoOfCUDA)

---

## 开篇：为什么GPU内存如此重要？

在上一篇中，我们学习了Grid、Block、Thread的层级结构。但有一个关键问题：

**即使你有10,000个核心并行计算，如果数据传输跟不上，性能依然会很差！**

一个真实的例子：
```cuda
// 看起来很简单的kernel
__global__ void simple_add(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = data[i] + 1.0f;  // 只是加1
    }
}
```

**性能分析**：
- 计算：1次浮点加法 = **1个时钟周期**
- 内存访问：从Global Memory读取 = **200-400个时钟周期**

**结论**：这个kernel有99%的时间在等待内存！

本文将深入讲解GPU的内存层级，以及如何优化内存访问。

---

## 1. GPU内存层级：金字塔结构

### 1.1 内存层级概览

![](https://raw.githubusercontent.com/Mark930209/MarkPicRepo/main/imgs/企业微信截图_17721723622030.png)

### 1.2 不同GPU架构的内存参数

| 架构 | GPU | Global Memory带宽 | L2 Cache | Shared Memory/SM | 寄存器/SM |
|------|-----|------------------|----------|-----------------|----------|
| **Ampere** | A100 | 1.5 TB/s (HBM2) | 40 MB | 164 KB | 256 KB |
| **Ampere** | RTX 3090 | 936 GB/s (GDDR6X) | 6 MB | 100 KB | 256 KB |
| **Hopper** | H100 | 3.35 TB/s (HBM3) | 50 MB | 228 KB | 256 KB |
| **Blackwell** | B200 | 8 TB/s (HBM3e) | 192 MB | 256 KB | 512 KB |

### 1.3 内存访问的性能差异

假设要访问1个float（4字节）：

| 内存类型 | 延迟 | 带宽 | 相对速度 |
|---------|------|------|---------|
| **寄存器** | 1 cycle | ~300 TB/s | 1× (基准) |
| **Shared Memory** | ~20 cycles | ~19 TB/s | 20× 慢 |
| **L1 Cache** | ~30 cycles | ~19 TB/s | 30× 慢 |
| **L2 Cache** | ~200 cycles | ~7 TB/s | 200× 慢 |
| **Global Memory** | 200-400 cycles | ~1.5 TB/s | 300× 慢 |
| **Host Memory** | 数千cycles | ~20 GB/s | 10,000× 慢 |

**关键洞察**：内存层级的速度差异是**数百倍**，优化内存访问比优化计算更重要！

---

## 2. Global Memory：GPU的主存储

### 2.1 Global Memory的特点

**优点**：
- ✅ 容量大（8-80 GB）
- ✅ 所有Thread都能访问
- ✅ 生命周期长（kernel之间持久）

**缺点**：
- ❌ 延迟高（200-400 cycles）
- ❌ 带宽有限（1.5-3.35 TB/s）
- ❌ 需要合并访问才能达到峰值带宽

### 2.2 Global Memory的分配与使用

```cuda
// 主机代码：分配Global Memory
float *d_data;
cudaMalloc(&d_data, n * sizeof(float));

// 拷贝数据到GPU
cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);

// Kernel中访问
__global__ void kernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = data[i] * 2.0f;  // 读取和写入Global Memory
    }
}

// 拷贝结果回CPU
cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);

// 释放
cudaFree(d_data);
```

### 2.3 Coalesced Access（合并访问）

**什么是合并访问？**

一个Warp（32个Thread）同时访问内存时，如果访问的地址是连续的，GPU可以将多个访问合并为一次内存事务。

**示例1：完美合并访问**
```cuda
__global__ void coalesced_access(float* data) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float value = data[i];  // ✅ 连续访问
}

// Warp 0的32个Thread访问：
// Thread 0: data[0]
// Thread 1: data[1]
// Thread 2: data[2]
// ...
// Thread 31: data[31]
// → 合并为1次128字节的内存事务（32×4字节）
```

**示例2：非合并访问**
```cuda
__global__ void uncoalesced_access(float* data) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float value = data[i * 32];  // ❌ 跨步访问
}

// Warp 0的32个Thread访问：
// Thread 0: data[0]
// Thread 1: data[32]
// Thread 2: data[64]
// ...
// Thread 31: data[992]
// → 需要32次内存事务！
```

**性能对比**：
```
合并访问:   1次内存事务 → 带宽利用率 100%
非合并访问: 32次内存事务 → 带宽利用率 3.125%
```

### 2.4 对齐要求

为了实现合并访问，地址需要对齐：

| 数据类型 | 大小 | 对齐要求 |
|---------|------|---------|
| `char` | 1字节 | 1字节 |
| `short` | 2字节 | 2字节 |
| `int`, `float` | 4字节 | 4字节 |
| `double`, `long long` | 8字节 | 8字节 |
| `float2` | 8字节 | 8字节 |
| `float4` | 16字节 | 16字节 |

**最佳实践**：
- 使用`cudaMalloc`分配的内存自动256字节对齐
- 结构体成员按大小排序，避免padding

---

## 3. Shared Memory：Block内的高速缓存

### 3.1 Shared Memory的特点

**优点**：
- ✅ 速度快（~19 TB/s，比Global Memory快10倍）
- ✅ 延迟低（~20 cycles）
- ✅ 程序员可控（显式管理）

**缺点**：
- ❌ 容量小（48-228 KB/Block）
- ❌ 只在Block内共享
- ❌ 需要手动管理（加载、同步）

### 3.2 Shared Memory的声明与使用

**静态分配**：
```cuda
__global__ void kernel_static() {
    __shared__ float shared_data[256];  // 编译时确定大小
    
    int tid = threadIdx.x;
    shared_data[tid] = tid;  // 每个Thread写入
    __syncthreads();         // 同步
    
    float value = shared_data[(tid + 1) % 256];  // 读取其他Thread的数据
}
```

**动态分配**：
```cuda
__global__ void kernel_dynamic(int size) {
    extern __shared__ float shared_data[];  // 运行时确定大小
    
    int tid = threadIdx.x;
    if (tid < size) {
        shared_data[tid] = tid;
    }
    __syncthreads();
}

// 启动时指定Shared Memory大小
kernel_dynamic<<<num_blocks, threads_per_block, shared_mem_bytes>>>(size);
```

### 3.3 Bank Conflict（存储体冲突）

**什么是Bank？**

Shared Memory被分为32个Bank（存储体），每个Bank每个时钟周期可以服务一个访问。

```
Bank 0: 地址 0, 32, 64, 96, ...
Bank 1: 地址 1, 33, 65, 97, ...
Bank 2: 地址 2, 34, 66, 98, ...
...
Bank 31: 地址 31, 63, 95, 127, ...
```

**无冲突访问**：
```cuda
__shared__ float data[256];

// ✅ 无冲突：每个Thread访问不同的Bank
int tid = threadIdx.x;
float value = data[tid];  // Thread 0→Bank 0, Thread 1→Bank 1, ...
```

**2-way冲突**：
```cuda
__shared__ float data[256];

// ❌ 2-way冲突：两个Thread访问同一个Bank
int tid = threadIdx.x;
float value = data[tid * 2];  // Thread 0,1→Bank 0, Thread 2,3→Bank 2, ...
// 性能降低2倍
```

**32-way冲突**（最坏情况）：
```cuda
__shared__ float data[256];

// ❌ 32-way冲突：所有Thread访问同一个Bank
int tid = threadIdx.x;
float value = data[0];  // 所有Thread都访问Bank 0
// 性能降低32倍！
```

**避免Bank Conflict的技巧**：

1. **Padding**：
```cuda
// ❌ 有冲突
__shared__ float data[32][32];
float value = data[threadIdx.x][0];  // 所有Thread访问列0 → 32-way冲突

// ✅ 无冲突：添加padding
__shared__ float data[32][33];  // 多一列
float value = data[threadIdx.x][0];  // 现在每个Thread访问不同的Bank
```

2. **交错访问**：
```cuda
// ❌ 有冲突
float value = data[threadIdx.x * 2];

// ✅ 无冲突
float value = data[threadIdx.x];
```

### 3.4 Shared Memory的典型应用：Tiling

**问题**：矩阵乘法 C = A × B

```cuda
// Naive版本：每个Thread从Global Memory读取N次
__global__ void matmul_naive(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];  // 每次循环读取2个Global Memory
    }
    C[row * N + col] = sum;
}
// 每个Thread读取Global Memory: 2N次
// 总读取次数: 2N³
```

**优化版本：使用Shared Memory Tiling**

```cuda
#define TILE_SIZE 16

__global__ void matmul_tiled(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // 分块计算
    for (int t = 0; t < N / TILE_SIZE; t++) {
        // 协作加载Tile到Shared Memory
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        __syncthreads();
        
        // 使用Shared Memory计算
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    C[row * N + col] = sum;
}
// 每个Thread读取Global Memory: 2N/TILE_SIZE次
// 总读取次数: 2N³/TILE_SIZE
// 加速比: TILE_SIZE倍（16倍！）
```

---

## 4. 寄存器：最快的存储

### 4.1 寄存器的特点

**优点**：
- ✅ 速度最快（1 cycle延迟）
- ✅ 带宽最高（~300 TB/s）
- ✅ 自动管理（编译器分配）

**缺点**：
- ❌ 数量有限（32-255个/Thread）
- ❌ Thread私有（不能共享）
- ❌ 溢出到Local Memory（很慢）

### 4.2 寄存器的使用

```cuda
__global__ void kernel() {
    // 局部变量自动分配到寄存器
    int x = threadIdx.x;        // 寄存器
    float y = x * 2.0f;         // 寄存器
    float z = y + 1.0f;         // 寄存器
    
    // 数组可能分配到寄存器（如果大小小且编译器能展开）
    float arr[4];               // 可能在寄存器
    arr[0] = 1.0f;
    arr[1] = 2.0f;
    arr[2] = 3.0f;
    arr[3] = 4.0f;
}
```

### 4.3 寄存器溢出（Register Spilling）

**问题**：如果一个Thread使用的寄存器超过限制，编译器会将部分变量"溢出"到Local Memory（实际在Global Memory中）。

```cuda
__global__ void register_heavy() {
    float a[100];  // 需要100个寄存器
    // 如果寄存器不够，部分数组元素会溢出到Local Memory
    // 性能大幅下降！
}
```

**查看寄存器使用**：
```bash
nvcc --ptxas-options=-v kernel.cu

# 输出：
# ptxas info : Used 32 registers, 0 bytes smem, 384 bytes cmem[0]
```

**优化策略**：
1. 减少局部变量
2. 使用`__launch_bounds__`限制寄存器使用
3. 增加Occupancy（让更多Warp隐藏延迟）

```cuda
// 限制每个Block最多256个Thread，每个Thread最多32个寄存器
__global__ void __launch_bounds__(256, 8) kernel() {
    // ...
}
```

---

## 5. Constant Memory与Texture Memory

### 5.1 Constant Memory

**特点**：
- 只读
- 64 KB大小
- 有专用缓存（8 KB/SM）
- 适合所有Thread读取相同数据

**使用场景**：
```cuda
// 在kernel外声明
__constant__ float const_data[1024];

// 主机代码：拷贝到Constant Memory
cudaMemcpyToSymbol(const_data, h_data, sizeof(float) * 1024);

// Kernel中使用
__global__ void kernel() {
    float value = const_data[10];  // 所有Thread读取相同值 → 广播
}
```

**性能**：
- 如果所有Thread读取相同地址：1次内存访问（广播）
- 如果Thread读取不同地址：串行化（性能差）

### 5.2 Texture Memory

**特点**：
- 只读
- 有专用缓存
- 支持硬件插值（图像处理）
- 支持边界处理

**使用场景**：图像处理、体数据渲染

```cuda
// 声明Texture对象
texture<float, 2, cudaReadModeElementType> tex;

// 绑定Texture
cudaBindTexture2D(NULL, tex, d_data, width, height, pitch);

// Kernel中使用
__global__ void kernel() {
    float value = tex2D(tex, x, y);  // 硬件插值
}
```

---

## 6. 内存访问模式优化

### 6.1 矩阵转置：经典案例

**Naive版本**：
```cuda
__global__ void transpose_naive(float* in, float* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // 读取：合并访问 ✅
        // 写入：非合并访问 ❌（列优先）
        out[x * height + y] = in[y * width + x];
    }
}
// 性能：~50 GB/s（只有峰值的3%）
```

**优化版本：使用Shared Memory**
```cuda
#define TILE_SIZE 32

__global__ void transpose_optimized(float* in, float* out, int width, int height) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];  // +1避免Bank Conflict
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // 读取到Shared Memory（合并访问）
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    }
    __syncthreads();
    
    // 转置后的坐标
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    // 从Shared Memory写入（合并访问）
    if (x < height && y < width) {
        out[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}
// 性能：~600 GB/s（接近峰值！）
```

### 6.2 Reduction：规约操作

**问题**：计算数组的和

**Naive版本**：
```cuda
__global__ void reduce_naive(float* data, float* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(result, data[i]);  // ❌ 原子操作串行化
    }
}
// 性能：非常差（所有Thread竞争一个地址）
```

**优化版本：使用Shared Memory**
```cuda
__global__ void reduce_optimized(float* data, float* result, int n) {
    __shared__ float shared_data[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 加载到Shared Memory
    shared_data[tid] = (i < n) ? data[i] : 0.0f;
    __syncthreads();
    
    // 树形规约
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    // Block的结果写回
    if (tid == 0) {
        atomicAdd(result, shared_data[0]);
    }
}
// 性能：快数百倍
```

---

## 7. 内存优化最佳实践

### 7.1 优化清单

| 优化项 | 说明 | 预期提升 |
|--------|------|---------|
| **合并访问** | 确保Warp内Thread访问连续地址 | 10-30× |
| **使用Shared Memory** | 缓存频繁访问的数据 | 5-20× |
| **避免Bank Conflict** | Padding或改变访问模式 | 2-32× |
| **减少寄存器使用** | 避免寄存器溢出 | 2-5× |
| **使用向量化加载** | `float4`代替`float` | 1.5-2× |
| **异步拷贝** | Overlap计算与传输 | 1.5-3× |

### 7.2 内存带宽测试

```cuda
__global__ void bandwidth_test(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = data[i] * 2.0f;  // 读+写
    }
}

// 计算带宽
float time_ms = /* 测量时间 */;
float bandwidth_GB_s = (n * sizeof(float) * 2) / (time_ms / 1000.0) / 1e9;
printf("带宽: %.2f GB/s\n", bandwidth_GB_s);
```

### 7.3 Roofline Model

**Roofline Model**用于分析kernel是计算受限还是内存受限：

```
性能 (GFLOPS)
    ↑
    │         ╱────────────  计算峰值
    │       ╱
    │     ╱  内存受限区域
    │   ╱
    │ ╱
    └──────────────────────→ 计算强度 (FLOPS/Byte)
```

**计算强度**：
```
计算强度 = 浮点运算数 / 内存访问字节数

示例1：向量加法
  c[i] = a[i] + b[i]
  运算: 1次加法 = 1 FLOP
  内存: 读2个float + 写1个float = 12字节
  计算强度 = 1 / 12 = 0.083 FLOPS/Byte
  → 内存受限

示例2：矩阵乘法（Tiled）
  C = A × B (N×N)
  运算: 2N³ FLOP
  内存: 3N² × 4字节
  计算强度 = 2N³ / (12N²) = N/6 FLOPS/Byte
  → N=1024时，计算强度=170，计算受限
```

---

## 总结与思考题

### 核心要点回顾

1. **内存层级**：寄存器 > Shared Memory > L1/L2 Cache > Global Memory，速度差异数百倍
2. **合并访问**：Warp内Thread访问连续地址，带宽利用率提升30倍
3. **Shared Memory**：Block内高速缓存，需要手动管理和同步
4. **Bank Conflict**：避免多个Thread访问同一个Bank，使用Padding技巧
5. **优化策略**：Tiling、Reduction、向量化加载

### 思考题

1. **为什么矩阵转置需要Shared Memory？** 提示：读写访问模式
2. **如何判断一个kernel是计算受限还是内存受限？** 提示：Roofline Model
3. **什么情况下应该使用Constant Memory？** 提示：访问模式

### 动手练习

1. 实现矩阵转置的Naive和优化版本，对比性能
2. 实现数组求和的Reduction，测试不同Block大小的性能
3. 使用`nvprof`或Nsight Compute分析内存带宽利用率

---

## 参考资料

[CUDA-PG-13.1] NVIDIA. "CUDA C++ Programming Guide v13.1 - Memory Hierarchy". https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy

[CUDA-BP-13.1] NVIDIA. "CUDA C++ Best Practices Guide - Memory Optimizations". https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations

[Harris2013-Shared] Harris, M. (2013). "Using Shared Memory in CUDA C/C++". NVIDIA Developer Blog.

[Kirk2022-PMPP] Kirk, D. B., & Hwu, W. W. (2022). "Programming Massively Parallel Processors" (4th ed.). Chapter 4: Memory Architecture.

---

## 下期预告

**第5篇：⚡ 第一个CUDA程序：向量加法到矩阵乘法**

下一篇我们将实战：
- 完整的CUDA程序结构
- 从向量加法到矩阵乘法
- 错误处理与调试
- 性能测量与分析

准备好写出高性能的CUDA代码了吗？让我们下期见！👋

---

*本文是《CUDA修仙之路》系列的第4篇，共38篇*  
*最后更新：2026年2月27日*
