# 🔍 深入SIMT执行模型：Warp、分支与占用率

> **系列**：CUDA修仙之路  
> **难度**：⭐⭐⭐ (1-5星)  
> **前置知识**：第3篇 - CUDA编程模型，第5篇 - 第一个CUDA程序  
> **预计阅读时间**：20分钟  
> **配套代码**：[GitHub链接](https://github.com/Mark930209/DaoOfCUDA)

---

## 开篇：为什么你的GPU只用了10%的算力？

你写了一个看起来完美的CUDA kernel：
```cuda
__global__ void process(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = data[i] * 2.0f;
    }
}
```

但性能分析工具显示：
- ❌ **GPU利用率**：只有10%
- ❌ **Occupancy**：只有25%
- ❌ **Warp执行效率**：只有50%

**问题出在哪里？**

答案在于你不理解GPU的**SIMT执行模型**。本文将深入讲解：
1. Warp是什么，为什么重要
2. Warp Divergence如何拖慢性能
3. Occupancy如何影响性能
4. 如何优化Warp级性能

---

## 1. Warp：GPU执行的真正单元

### 1.1 从Thread到Warp

在第3篇中，我们学习了Thread是GPU执行的最小单元。但这只是**逻辑上**的，**物理上**GPU执行的基本单元是**Warp**。

![](https://raw.githubusercontent.com/Mark930209/MarkPicRepo/main/imgs/企业微信截图_17725061113742.png)

**Warp的定义**：
- 一个Warp包含**32个连续的Thread**
- 这32个Thread**同时执行相同的指令**
- Warp是SM调度的基本单元

### 1.2 Warp的划分规则

![](https://raw.githubusercontent.com/Mark930209/MarkPicRepo/main/imgs/企业微信截图_1772528287829.png)

**划分规则**：
- 按照`threadIdx.x`的顺序划分
- 每32个Thread一组
- 2D/3D Block也是按照线性化后的索引划分

**2D Block示例**：

![](https://raw.githubusercontent.com/Mark930209/MarkPicRepo/main/imgs/企业微信截图_17725283278369.png)

### 1.3 SIMT vs SIMD

| 特性 | SIMT (GPU) | SIMD (CPU) |
|------|-----------|-----------|
| **全称** | Single Instruction, Multiple Thread | Single Instruction, Multiple Data |
| **执行单元** | 32个Thread | 4-16个数据通道 |
| **分支处理** | 可以分支，但会串行化 | 通常不支持分支 |
| **编程模型** | 每个Thread独立编程 | 显式向量化 |
| **灵活性** | 高（可以有条件分支） | 低（必须所有通道执行相同操作） |

**SIMT的优势**：
```cuda
// SIMT：每个Thread可以有不同的控制流
if (threadIdx.x < 16) {
    // 前16个Thread执行这里
} else {
    // 后16个Thread执行这里
}
```

```c
// SIMD：所有通道必须执行相同操作
__m256 a = _mm256_load_ps(data);
__m256 b = _mm256_mul_ps(a, a);  // 所有8个float同时平方
```

---

## 2. Warp调度与执行

### 2.1 SM的Warp调度器

每个SM有多个Warp调度器（Scheduler）：

| 架构 | Warp调度器/SM | 每周期可发射指令 |
|------|--------------|----------------|
| **Volta** | 4 | 4条指令 |
| **Ampere** | 4 | 4条指令 |
| **Hopper** | 4 | 4条指令 |
| **Blackwell** | 4 | 4条指令 |

**调度策略**：
```
时钟周期1: Warp 0执行指令1
时钟周期2: Warp 1执行指令1（Warp 0等待内存）
时钟周期3: Warp 2执行指令1
时钟周期4: Warp 3执行指令1
...
时钟周期10: Warp 0执行指令2（内存已就绪）
```

**关键洞察**：通过快速切换Warp，隐藏内存延迟！

### 2.2 Warp的状态

一个Warp在任意时刻处于以下状态之一：

| 状态 | 说明 | 原因 |
|------|------|------|
| **Active** | 正在执行 | 被调度器选中 |
| **Stalled** | 等待中 | 等待内存、等待同步、等待依赖 |
| **Eligible** | 可执行 | 准备好执行，等待调度 |
| **Completed** | 已完成 | 执行完所有指令 |

**Stall的原因**：
1. **Memory Stall**：等待Global Memory访问（200-400 cycles）
2. **Sync Stall**：等待`__syncthreads()`
3. **Execution Dependency**：等待前一条指令的结果
4. **Texture Stall**：等待Texture Memory访问

### 2.3 隐藏延迟的原理

**问题**：Global Memory访问需要400个时钟周期，如何不浪费这些时间？

**答案**：运行足够多的Warp！

```
假设：
- Global Memory延迟：400 cycles
- 每个Warp每条指令：1 cycle
- 需要多少个Warp才能隐藏延迟？

答案：至少400个Warp！

但实际上：
- 每个SM最多支持32-64个Warp
- 所以需要多个SM协同工作
```

**示例**：
```
SM有32个活跃Warp：

Cycle 1:   Warp 0发起内存访问，进入Stalled状态
Cycle 2:   Warp 1执行计算
Cycle 3:   Warp 2执行计算
...
Cycle 32:  Warp 31执行计算
Cycle 33:  Warp 0执行计算（如果内存已就绪）
           或者继续执行Warp 1-31
```

---

## 3. Warp Divergence：性能杀手

### 3.1 什么是Warp Divergence？

当一个Warp内的Thread执行不同的代码路径时，就会发生**Warp Divergence**（分支分化）。

**示例**：
```cuda
__global__ void divergent_kernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i % 2 == 0) {
        // 偶数Thread执行这里
        data[i] = data[i] * 2.0f;
    } else {
        // 奇数Thread执行这里
        data[i] = data[i] + 1.0f;
    }
}
```

**执行过程**：
```
Warp 0 (Thread 0-31):
  Step 1: Thread 0,2,4,...,30 执行 data[i] * 2.0f
          Thread 1,3,5,...,31 等待（Inactive）
  
  Step 2: Thread 1,3,5,...,31 执行 data[i] + 1.0f
          Thread 0,2,4,...,30 等待（Inactive）

总时间 = 2倍（串行执行）
```

### 3.2 Divergence的性能影响

**性能损失**：
```
无Divergence: 1个Warp执行1次
有Divergence: 1个Warp执行N次（N = 不同路径数）

最坏情况：32个Thread走32条不同路径 → 性能降低32倍！
```

**实际测试**：
```cuda
// 测试1：无Divergence
__global__ void no_divergence(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = data[i] * 2.0f;  // 所有Thread执行相同操作
    }
}
// 性能：100%

// 测试2：2-way Divergence
__global__ void two_way_divergence(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i % 2 == 0) {
        data[i] = data[i] * 2.0f;
    } else {
        data[i] = data[i] + 1.0f;
    }
}
// 性能：50%（2倍慢）

// 测试3：32-way Divergence
__global__ void worst_divergence(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i % 32 == threadIdx.x % 32) {
        data[i] = data[i] * 2.0f;
    }
}
// 性能：3%（32倍慢）
```

### 3.3 避免Divergence的技巧

#### 技巧1：重新组织数据

**问题代码**：
```cuda
// 奇偶交替存储：[偶, 奇, 偶, 奇, ...]
__global__ void bad(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i % 2 == 0) {
        data[i] = process_even(data[i]);
    } else {
        data[i] = process_odd(data[i]);
    }
}
// Divergence：每个Warp都有分支
```

**优化代码**：
```cuda
// 分块存储：[偶偶偶..., 奇奇奇...]
__global__ void good(float* even_data, float* odd_data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (blockIdx.x % 2 == 0) {
        // 整个Block处理偶数
        even_data[i] = process_even(even_data[i]);
    } else {
        // 整个Block处理奇数
        odd_data[i] = process_odd(odd_data[i]);
    }
}
// 无Divergence：每个Warp执行相同路径
```

#### 技巧2：使用Warp级函数

**问题代码**：
```cuda
__global__ void bad_reduction(float* data, float* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(result, data[i]);  // 所有Thread竞争
    }
}
```

**优化代码**：
```cuda
__global__ void good_reduction(float* data, float* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float value = (i < n) ? data[i] : 0.0f;
    
    // Warp内规约（无Divergence）
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    
    // 只有Warp的第一个Thread写入
    if (threadIdx.x % 32 == 0) {
        atomicAdd(result, value);
    }
}
```

#### 技巧3：使用谓词执行

**问题代码**：
```cuda
if (condition) {
    result = a + b;
} else {
    result = a - b;
}
```

**优化代码**：
```cuda
// 使用三元运算符（编译器可能优化为谓词执行）
result = condition ? (a + b) : (a - b);
```

### 3.4 不可避免的Divergence

有些情况下Divergence无法避免：

```cuda
// 边界检查（必需）
if (i < n) {
    data[i] = ...;
}

// 稀疏矩阵（数据相关）
if (matrix[i] != 0) {
    result += matrix[i] * vector[i];
}
```

**解决方案**：
1. 确保大部分Warp不会Divergence（只有最后几个Warp有边界问题）
2. 使用更大的Block（减少边界Warp的比例）
3. 接受性能损失（有时无法避免）

---

## 4. Occupancy：SM的利用率

### 4.1 Occupancy的定义

**Occupancy（占用率）**：
```
Occupancy = (活跃Warp数) / (最大Warp数)
```

**示例**：
```
假设：
- 每个SM最多支持64个Warp
- 当前SM上有32个活跃Warp

Occupancy = 32 / 64 = 50%
```

### 4.2 影响Occupancy的因素

#### 因素1：Block大小

```cuda
// 示例：每个SM最多64个Warp = 2048个Thread

// 配置1：Block大小 = 1024
// 每个Block = 32个Warp
// 每个SM可以运行：2048 / 1024 = 2个Block = 64个Warp
// Occupancy = 100% ✅

// 配置2：Block大小 = 512
// 每个Block = 16个Warp
// 每个SM可以运行：2048 / 512 = 4个Block = 64个Warp
// Occupancy = 100% ✅

// 配置3：Block大小 = 768
// 每个Block = 24个Warp
// 每个SM可以运行：2048 / 768 = 2个Block = 48个Warp
// Occupancy = 75% ❌（浪费资源）
```

#### 因素2：寄存器使用

```cuda
// 假设：
// - 每个SM有65536个寄存器
// - Block大小 = 256个Thread

// Kernel 1：每个Thread使用32个寄存器
// 每个Block需要：256 × 32 = 8192个寄存器
// 每个SM可以运行：65536 / 8192 = 8个Block
// Occupancy = 8 × 8 / 64 = 100% ✅

// Kernel 2：每个Thread使用64个寄存器
// 每个Block需要：256 × 64 = 16384个寄存器
// 每个SM可以运行：65536 / 16384 = 4个Block
// Occupancy = 4 × 8 / 64 = 50% ❌
```

**查看寄存器使用**：
```bash
nvcc --ptxas-options=-v kernel.cu

# 输出：
# ptxas info : Used 32 registers, 4096 bytes smem
```

#### 因素3：Shared Memory使用

```cuda
// 假设：
// - 每个SM有164 KB Shared Memory
// - Block大小 = 256个Thread

// Kernel 1：每个Block使用16 KB Shared Memory
// 每个SM可以运行：164 / 16 = 10个Block
// Occupancy = 10 × 8 / 64 = 125% → 限制为100% ✅

// Kernel 2：每个Block使用48 KB Shared Memory
// 每个SM可以运行：164 / 48 = 3个Block
// Occupancy = 3 × 8 / 64 = 37.5% ❌
```

#### 因素4：Block数量限制

每个SM对Block数量有硬件限制：

| 架构 | 最大Block数/SM |
|------|---------------|
| **Volta** | 32 |
| **Ampere** | 32 |
| **Hopper** | 32 |

```cuda
// 如果Block太小：
// Block大小 = 32个Thread = 1个Warp
// 即使有足够的资源，每个SM最多32个Block = 32个Warp
// Occupancy = 32 / 64 = 50% ❌
```

### 4.3 Occupancy Calculator

**NVIDIA提供的工具**：
```bash
# 在线计算器
https://docs.nvidia.com/cuda/cuda-occupancy-calculator/

# 或使用API
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, kernel, blockSize, dynamicSMem);
```

**示例**：
```cuda
int blockSize = 256;
int minGridSize, gridSize;

// 自动选择最优Block大小
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0);

printf("推荐Block大小: %d\n", blockSize);
```

### 4.4 Occupancy越高越好吗？

**答案：不一定！**

```
高Occupancy的好处：
✅ 更多Warp可以隐藏延迟
✅ 更好的SM利用率

高Occupancy的代价：
❌ 每个Thread可用的寄存器更少
❌ 每个Block可用的Shared Memory更少
❌ 可能导致寄存器溢出（Register Spilling）

最佳Occupancy：通常50-75%就足够了
```

**实际案例**：
```
Kernel A：
- Occupancy = 100%
- 每个Thread使用16个寄存器
- 性能：500 GFLOPS

Kernel B：
- Occupancy = 50%
- 每个Thread使用64个寄存器（更多优化）
- 性能：800 GFLOPS ✅（更快！）
```

---

## 5. 优化Warp级性能

### 5.1 优化清单

| 优化项 | 目标 | 方法 |
|--------|------|------|
| **避免Divergence** | 减少分支分化 | 重组数据、使用Warp函数 |
| **提高Occupancy** | 增加活跃Warp | 调整Block大小、减少资源使用 |
| **减少Warp Stall** | 减少等待时间 | 优化内存访问、使用异步操作 |
| **提高ILP** | 增加指令级并行 | 展开循环、多个独立操作 |

### 5.2 使用Warp级原语

CUDA提供了Warp级函数，无需同步：

```cuda
// Warp Shuffle：在Warp内交换数据
__shfl_sync(mask, var, srcLane);      // 从指定Lane读取
__shfl_up_sync(mask, var, delta);     // 从上方Lane读取
__shfl_down_sync(mask, var, delta);   // 从下方Lane读取
__shfl_xor_sync(mask, var, laneMask); // XOR模式交换

// Warp Vote：在Warp内投票
__all_sync(mask, predicate);   // 所有Thread都为true？
__any_sync(mask, predicate);   // 任意Thread为true？
__ballot_sync(mask, predicate); // 返回每个Thread的投票结果

// Warp Match：查找相同值的Thread
__match_any_sync(mask, value);  // 找到值相同的Thread
__match_all_sync(mask, value);  // 所有Thread值相同？
```

**示例：Warp Reduce**
```cuda
__device__ float warp_reduce_sum(float value) {
    // 无需__syncthreads()，Warp内自动同步
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__ void fast_reduction(float* data, float* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float value = (i < n) ? data[i] : 0.0f;
    
    // Warp级规约
    value = warp_reduce_sum(value);
    
    // 每个Warp的第一个Thread写入
    if (threadIdx.x % 32 == 0) {
        atomicAdd(result, value);
    }
}
```

### 5.3 循环展开

**问题代码**：
```cuda
for (int i = 0; i < 4; i++) {
    sum += data[i];
}
// 4次循环，4次分支判断
```

**优化代码**：
```cuda
#pragma unroll
for (int i = 0; i < 4; i++) {
    sum += data[i];
}
// 编译器展开为：
// sum += data[0];
// sum += data[1];
// sum += data[2];
// sum += data[3];
// 无循环开销，更高的ILP
```

### 5.4 使用`__launch_bounds__`

```cuda
// 限制每个Block最多256个Thread，每个Thread最多32个寄存器
__global__ void __launch_bounds__(256, 8) optimized_kernel() {
    // ...
}
```

**参数说明**：
- 第一个参数：maxThreadsPerBlock
- 第二个参数：minBlocksPerMultiprocessor（可选）

**效果**：
- 编译器会优化寄存器使用
- 保证至少指定数量的Block可以运行

---

## 6. 性能分析工具

### 6.1 Nsight Compute指标

| 指标 | 说明 | 目标 |
|------|------|------|
| **Warp Execution Efficiency** | Warp执行效率 | >90% |
| **Occupancy** | 占用率 | 50-100% |
| **Branch Efficiency** | 分支效率 | >90% |
| **Stall Reasons** | Warp停顿原因 | 分析瓶颈 |

**使用方法**：
```bash
ncu --set full -o profile ./program

# 查看报告
ncu-ui profile.ncu-rep
```

### 6.2 关键指标解读

**Warp Execution Efficiency**：
```
Warp Execution Efficiency = (活跃Thread数) / (总Thread数)

示例：
- 32个Thread的Warp
- 由于Divergence，平均只有16个Thread活跃
- Efficiency = 16 / 32 = 50%
```

**Branch Efficiency**：
```
Branch Efficiency = (不发生Divergence的分支数) / (总分支数)

示例：
- 100个分支
- 10个发生Divergence
- Efficiency = 90 / 100 = 90%
```

---

## 7. 实战案例：优化直方图计算

### 7.1 Naive版本

```cuda
__global__ void histogram_naive(int* data, int* hist, int n, int bins) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        int bin = data[i] % bins;
        atomicAdd(&hist[bin], 1);  // 所有Thread竞争
    }
}
// 问题：原子操作串行化，Divergence严重
```

### 7.2 优化版本1：Shared Memory

```cuda
__global__ void histogram_shared(int* data, int* hist, int n, int bins) {
    __shared__ int local_hist[256];
    
    // 初始化
    if (threadIdx.x < bins) {
        local_hist[threadIdx.x] = 0;
    }
    __syncthreads();
    
    // 计算局部直方图
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int bin = data[i] % bins;
        atomicAdd(&local_hist[bin], 1);  // Block内竞争
    }
    __syncthreads();
    
    // 合并到全局直方图
    if (threadIdx.x < bins) {
        atomicAdd(&hist[threadIdx.x], local_hist[threadIdx.x]);
    }
}
// 改进：减少全局原子操作
```

### 7.3 优化版本2：Warp级聚合

```cuda
__global__ void histogram_warp(int* data, int* hist, int n, int bins) {
    __shared__ int local_hist[256];
    
    if (threadIdx.x < bins) {
        local_hist[threadIdx.x] = 0;
    }
    __syncthreads();
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int bin = data[i] % bins;
        
        // Warp内聚合
        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;
        
        // 使用Warp Vote检查是否有其他Thread访问相同bin
        unsigned int mask = __ballot_sync(0xffffffff, true);
        
        // 只有Warp内第一个访问该bin的Thread执行原子操作
        if (__popc(__ballot_sync(mask, bin == bin)) == 1) {
            atomicAdd(&local_hist[bin], 1);
        }
    }
    __syncthreads();
    
    if (threadIdx.x < bins) {
        atomicAdd(&hist[threadIdx.x], local_hist[threadIdx.x]);
    }
}
```

---

## 总结与思考题

### 核心要点回顾

1. **Warp**：32个Thread的组，是GPU物理执行的基本单元
2. **SIMT**：Single Instruction, Multiple Thread，比SIMD更灵活
3. **Warp Divergence**：分支分化导致串行执行，性能损失严重
4. **Occupancy**：活跃Warp数/最大Warp数，不是越高越好
5. **优化策略**：避免Divergence、提高Occupancy、使用Warp级函数

### 思考题

1. **为什么Warp大小是32？** 提示：硬件设计权衡
2. **如何判断一个kernel是否有严重的Divergence？** 提示：Nsight Compute
3. **Occupancy 50%和100%，哪个更好？** 提示：取决于kernel特性

### 动手练习

1. 编写一个有严重Divergence的kernel，测量性能损失
2. 使用Nsight Compute分析一个kernel的Occupancy
3. 实现Warp级的求和、求最大值

---

## 参考资料

[CUDA-PG-13.1] NVIDIA. "CUDA C++ Programming Guide v13.1 - SIMT Architecture". https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#simt-architecture

[CUDA-BP-13.1] NVIDIA. "CUDA C++ Best Practices Guide - Execution Configuration Optimizations". https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#execution-configuration-optimizations

[Jia2018-Volta] Jia, Z., et al. (2018). "Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking". arXiv:1804.06826.

[Kirk2022-PMPP] Kirk, D. B., & Hwu, W. W. (2022). "Programming Massively Parallel Processors" (4th ed.). Chapter 5: Performance Considerations.

---

## 下期预告

**第7篇：📊 性能分析实战：Nsight Systems与Nsight Compute**

下一篇我们将实战：
- Nsight Systems：Timeline分析
- Nsight Compute：Kernel详细分析
- Roofline Model：性能瓶颈诊断
- 实战案例：优化一个真实kernel

准备好成为性能调优专家了吗？让我们下期见！👋

---

*本文是《CUDA修仙之路》系列的第6篇，共38篇*  
*最后更新：2026年3月3日*
