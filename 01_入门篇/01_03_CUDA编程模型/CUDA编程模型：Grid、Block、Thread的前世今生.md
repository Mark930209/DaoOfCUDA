# 🧵 CUDA编程模型：Grid、Block、Thread的前世今生

> **系列**：CUDA修仙之路  
> **难度**：⭐⭐ (1-5星)  
> **前置知识**：第1篇 - GPU编程革命，第2篇 - 开发环境搭建  
> **预计阅读时间**：18分钟  
> **配套代码**：[GitHub链接](https://github.com/Mark930209/DaoOfCUDA)

---

## 开篇：为什么GPU需要三级线程层级？

在第1篇中，我们看到GPU有10,000+个核心。但问题来了：

**如果你要处理1亿个数据点，如何组织这10,000个核心的工作？**

- ❌ 让每个核心处理10,000个数据？（太慢，没有充分并行）
- ❌ 创建1亿个线程？（管理开销太大）
- ✅ **CUDA的答案**：三级层级结构 Grid → Block → Thread

这就像管理一个超大型工厂：
```
工厂（GPU）
├─ 车间1（Block 0）
│   ├─ 工人1（Thread 0）
│   ├─ 工人2（Thread 1）
│   └─ ...
├─ 车间2（Block 1）
│   ├─ 工人1（Thread 0）
│   └─ ...
└─ ...
```

本文将深入讲解这个设计的精妙之处。

---

## 1. CUDA线程层级：从宏观到微观

### 1.1 三级结构概览
![](https://raw.githubusercontent.com/Mark930209/MarkPicRepo/main/imgs/企业微信截图_17721726811277.png)

### 1.2 一个直观的例子：图像处理

假设要处理一张1920×1080的图像（约200万像素）：

```cuda
// 每个线程处理一个像素
__global__ void process_image(uchar4* image, int width, int height) {
    // 计算当前线程负责的像素坐标
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        // 处理像素 image[idx]
        image[idx].x = 255 - image[idx].x;  // 反色
    }
}

// 启动配置
dim3 threads_per_block(16, 16);  // 每个Block: 16×16=256个线程
dim3 num_blocks(
    (width + 15) / 16,   // X方向需要的Block数
    (height + 15) / 16   // Y方向需要的Block数
);

process_image<<<num_blocks, threads_per_block>>>(d_image, width, height);
```

**可视化**：
```
Grid: 120×68 个Block
每个Block: 16×16 个Thread
总线程数: 120×68×16×16 = 2,088,960 个线程

Block(0,0)          Block(1,0)          Block(2,0)
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│T T T ... T T│    │T T T ... T T│    │T T T ... T T│
│T T T ... T T│    │T T T ... T T│    │T T T ... T T│
│. . .     . .│    │. . .     . .│    │. . .     . .│
│T T T ... T T│    │T T T ... T T│    │T T T ... T T│
└─────────────┘    └─────────────┘    └─────────────┘
      ↓                  ↓                  ↓
   处理像素           处理像素           处理像素
   (0,0)-(15,15)     (16,0)-(31,15)    (32,0)-(47,15)
```

---

## 2. Thread：GPU执行的最小单元

### 2.1 Thread的身份标识

每个Thread通过三个内置变量知道自己是谁：

| 变量 | 类型 | 含义 | 范围 |
|------|------|------|------|
| `threadIdx.x/y/z` | `uint3` | Thread在Block中的索引 | 0 到 blockDim-1 |
| `blockIdx.x/y/z` | `uint3` | Block在Grid中的索引 | 0 到 gridDim-1 |
| `blockDim.x/y/z` | `dim3` | Block的维度（每个Block的Thread数） | 编译时常量 |
| `gridDim.x/y/z` | `dim3` | Grid的维度（Grid中的Block数） | 编译时常量 |

### 2.2 计算全局索引

**1D情况**（最常见）：
```cuda
int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

// 示例：blockDim.x = 256
// Block 0: Thread 0-255   → global_idx = 0-255
// Block 1: Thread 0-255   → global_idx = 256-511
// Block 2: Thread 0-255   → global_idx = 512-767
```

**2D情况**（图像处理）：
```cuda
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int global_idx = y * width + x;  // 行优先
```

**3D情况**（体数据处理）：
```cuda
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
int global_idx = z * (width * height) + y * width + x;
```

### 2.3 Thread的资源

每个Thread拥有：
- **寄存器**：最快的存储（~1 cycle延迟），但数量有限（32-255个，取决于架构）
- **局部内存**：寄存器溢出时使用，实际在Global Memory中（慢）
- **访问权限**：
  - ✅ 自己的寄存器和局部内存
  - ✅ Block内的Shared Memory
  - ✅ 全局的Global Memory
  - ✅ 只读的Constant Memory和Texture Memory

---

## 3. Block：协作的线程组

### 3.1 为什么需要Block？

**问题**：如果只有Thread，没有Block会怎样？

假设要计算矩阵乘法 C = A × B：
```
C[i][j] = Σ A[i][k] * B[k][j]
          k
```

每个Thread计算一个C[i][j]，需要：
1. 读取A的一行（N个元素）
2. 读取B的一列（N个元素）
3. 进行N次乘加运算

**问题**：如果N=1024，每个Thread都要从Global Memory读取2048个数据！

**Block的解决方案**：
- Block内的Thread共享Shared Memory（19 TB/s，比Global Memory快10倍）
- 协作加载数据到Shared Memory，然后共享使用
- 减少Global Memory访问次数

### 3.2 Block的特性

| 特性 | 说明 | 限制 |
|------|------|------|
| **大小限制** | 每个Block最多1024个Thread（Compute Capability ≥ 2.0） | 硬件限制 |
| **维度** | 可以是1D、2D或3D | blockDim.x × blockDim.y × blockDim.z ≤ 1024 |
| **Shared Memory** | Block内共享，Block间隔离 | 48-164 KB（取决于架构） |
| **同步** | Block内可以同步（`__syncthreads()`） | Block间不能同步 |
| **独立调度** | 每个Block独立调度到SM | Block间执行顺序不确定 |

### 3.3 Block大小的选择

**常见的Block大小**：

| 维度 | 常用配置 | 适用场景 |
|------|---------|---------|
| **1D** | 128, 256, 512 | 向量运算、Reduction |
| **2D** | 16×16, 32×32 | 图像处理、矩阵运算 |
| **3D** | 8×8×8 | 体数据处理、3D卷积 |

**选择原则**：
1. **32的倍数**：因为Warp大小是32（后面会讲）
2. **不要太小**：<128会导致SM利用率低
3. **不要太大**：>512可能导致寄存器/Shared Memory不足
4. **256是黄金值**：在大多数情况下表现良好

### 3.4 Block内同步：`__syncthreads()`

```cuda
__global__ void example_sync(float* data) {
    __shared__ float shared_data[256];
    
    int tid = threadIdx.x;
    
    // 阶段1：每个Thread加载数据到Shared Memory
    shared_data[tid] = data[blockIdx.x * blockDim.x + tid];
    
    // 同步：确保所有Thread都完成加载
    __syncthreads();
    
    // 阶段2：现在可以安全地访问其他Thread加载的数据
    float sum = shared_data[tid] + shared_data[(tid + 1) % blockDim.x];
    
    data[blockIdx.x * blockDim.x + tid] = sum;
}
```

**关键点**：
- `__syncthreads()` 是Block内的栅栏（barrier）
- 所有Thread必须都执行到这里才能继续
- ⚠️ **不能在条件分支中使用**（会导致死锁）

**错误示例**：
```cuda
if (threadIdx.x < 128) {
    __syncthreads();  // ❌ 错误！只有一半Thread会执行
}
```

**正确示例**：
```cuda
__syncthreads();  // ✅ 正确！所有Thread都会执行

if (threadIdx.x < 128) {
    // 条件代码
}
```

---

## 4. Grid：kernel的执行空间

### 4.1 Grid的配置

Grid定义了有多少个Block：

```cuda
// 1D Grid
int num_blocks = (n + threads_per_block - 1) / threads_per_block;
kernel<<<num_blocks, threads_per_block>>>(args);

// 2D Grid
dim3 grid(num_blocks_x, num_blocks_y);
dim3 block(threads_per_block_x, threads_per_block_y);
kernel<<<grid, block>>>(args);

// 3D Grid
dim3 grid(num_blocks_x, num_blocks_y, num_blocks_z);
dim3 block(threads_per_block_x, threads_per_block_y, threads_per_block_z);
kernel<<<grid, block>>>(args);
```

### 4.2 Grid的限制

| 架构 | gridDim.x | gridDim.y | gridDim.z | 总Block数 |
|------|-----------|-----------|-----------|----------|
| Compute Capability 2.x | 65535 | 65535 | 65535 | ~2^48 |
| Compute Capability 3.0+ | 2^31-1 | 65535 | 65535 | 巨大 |

**实际限制**：通常受GPU的SM数量和内存限制。

### 4.3 Grid的执行模型

**关键特性**：
1. **Block独立性**：Block之间不能通信、不能同步
2. **执行顺序不确定**：Block可能以任意顺序执行
3. **可扩展性**：同一个kernel可以在不同GPU上运行（自动适应SM数量）

**为什么Block独立？**

```
GPU with 10 SMs          GPU with 80 SMs
┌─────────────┐          ┌─────────────┐
│ SM0: B0,B10 │          │ SM0: B0     │
│ SM1: B1,B11 │          │ SM1: B1     │
│ ...         │          │ ...         │
│ SM9: B9,B19 │          │ SM79: B79   │
└─────────────┘          └─────────────┘
  同一个kernel             同一个kernel
  自动适应10个SM           自动适应80个SM
```

---

## 5. 硬件映射：软件模型到物理硬件

### 5.1 SM（Streaming Multiprocessor）

**SM是GPU的核心计算单元**：

| GPU型号 | SM数量 | 每个SM的CUDA Core | 总CUDA Core |
|---------|--------|------------------|-------------|
| RTX 3060 | 28 | 128 | 3584 |
| RTX 4090 | 128 | 128 | 16384 |
| A100 | 108 | 64 | 6912 |
| H100 | 132 | 128 | 16896 |

![](https://raw.githubusercontent.com/Mark930209/MarkPicRepo/main/imgs/企业微信截图_17722705168676.png)

**SM的结构**（以Ampere为例）：
```
SM
├─ 4个Processing Block
│   ├─ 16个FP32 CUDA Core
│   ├─ 16个INT32 Core
│   ├─ 8个FP64 Core
│   ├─ 1个Tensor Core
│   └─ Warp Scheduler
├─ 128 KB L1 Cache / Shared Memory
├─ 64K 32-bit寄存器
└─ 4个Texture Unit
```

### 5.2 Block到SM的映射

**调度规则**：
1. 每个Block完整地分配到一个SM
2. 一个SM可以同时运行多个Block（如果资源足够）
3. Block一旦开始执行，就会一直在该SM上直到完成

**资源限制**：
```
每个SM能运行的Block数 = min(
    硬件限制（通常16-32个），
    寄存器限制（总寄存器数 / 每个Block需要的寄存器数），
    Shared Memory限制（总Shared Memory / 每个Block需要的Shared Memory）
)
```

**示例**：
```
假设：
- SM有65536个寄存器
- 每个Block有256个Thread
- 每个Thread使用32个寄存器

每个Block需要的寄存器 = 256 × 32 = 8192
每个SM能运行的Block数 = 65536 / 8192 = 8个
```

### 5.3 Warp：硬件执行的基本单元

**Warp是32个连续Thread的组**：

```cuda
// Block有256个Thread
// 自动分为8个Warp：
Warp 0: Thread 0-31
Warp 1: Thread 32-63
Warp 2: Thread 64-95
...
Warp 7: Thread 224-255
```

**SIMT执行模型**：
- 一个Warp中的32个Thread执行相同的指令
- 但可以处理不同的数据（SIMT = Single Instruction, Multiple Thread）
- 类似于CPU的SIMD，但更灵活

**Warp调度**：
```
时钟周期1: Warp 0执行指令1
时钟周期2: Warp 1执行指令1（Warp 0等待内存）
时钟周期3: Warp 2执行指令1
...
时钟周期9: Warp 0执行指令2（内存已就绪）
```

**隐藏延迟的关键**：当一个Warp等待内存时，切换到另一个Warp执行。

---

## 6. 实战：不同维度的Grid/Block配置

### 6.1 案例1：1D向量加法

```cuda
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// 配置
int n = 10000000;
int threads_per_block = 256;
int num_blocks = (n + threads_per_block - 1) / threads_per_block;

vector_add<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, n);
```

**可视化**：
```
n = 10,000,000
threads_per_block = 256
num_blocks = 39063

Grid: [39063]
Block: [256]

Block 0: Thread 0-255     → 处理元素 0-255
Block 1: Thread 0-255     → 处理元素 256-511
...
Block 39062: Thread 0-255 → 处理元素 9,999,872-10,000,127
```

### 6.2 案例2：2D矩阵加法

```cuda
__global__ void matrix_add(float* a, float* b, float* c, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        int idx = row * width + col;
        c[idx] = a[idx] + b[idx];
    }
}

// 配置
int width = 1024, height = 1024;
dim3 threads_per_block(16, 16);  // 256个Thread
dim3 num_blocks(
    (width + 15) / 16,   // 64
    (height + 15) / 16   // 64
);

matrix_add<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, width, height);
```

**可视化**：
```
Matrix: 1024×1024
Block: 16×16 (256 threads)
Grid: 64×64 (4096 blocks)

Block(0,0)处理: 像素(0,0)到(15,15)
Block(1,0)处理: 像素(16,0)到(31,15)
Block(0,1)处理: 像素(0,16)到(15,31)
...
```

### 6.3 案例3：3D体数据处理

```cuda
__global__ void volume_process(float* volume, int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < width && y < height && z < depth) {
        int idx = z * (width * height) + y * width + x;
        volume[idx] = volume[idx] * 2.0f;  // 示例操作
    }
}

// 配置
int width = 256, height = 256, depth = 256;
dim3 threads_per_block(8, 8, 8);  // 512个Thread
dim3 num_blocks(
    (width + 7) / 8,   // 32
    (height + 7) / 8,  // 32
    (depth + 7) / 8    // 32
);

volume_process<<<num_blocks, threads_per_block>>>(d_volume, width, height, depth);
```

---

## 7. 性能考虑：如何选择Grid/Block配置

### 7.1 Occupancy（占用率）

**定义**：活跃Warp数 / 最大Warp数

```
Occupancy = (每个SM的活跃Warp数) / (每个SM的最大Warp数)
```

**目标**：通常希望Occupancy ≥ 50%

**影响因素**：
1. **Block大小**：太小导致Warp数不足，太大导致资源不足
2. **寄存器使用**：每个Thread用的寄存器越多，能运行的Thread越少
3. **Shared Memory使用**：每个Block用的Shared Memory越多，能运行的Block越少

**NVIDIA Occupancy Calculator**：
```bash
# 使用nvcc编译时查看资源使用
nvcc --ptxas-options=-v kernel.cu

# 输出示例：
# ptxas info : Used 32 registers, 4096 bytes smem, 384 bytes cmem[0]
```

### 7.2 最佳实践

| 原则 | 说明 | 示例 |
|------|------|------|
| **Block大小是32的倍数** | 避免Warp内有空闲Thread | 128, 256, 512 |
| **优先选择256** | 在大多数情况下表现良好 | threads_per_block = 256 |
| **2D Block用16×16或32×32** | 平衡X/Y维度 | dim3(16, 16) |
| **避免太小的Block** | <128会导致Occupancy低 | 不要用64 |
| **避免太大的Block** | >512可能资源不足 | 不要用1024 |
| **Grid要足够大** | 至少是SM数的2-4倍 | num_blocks ≥ SM_count × 4 |

### 7.3 边界检查的重要性

**为什么需要边界检查？**

```cuda
// 假设n = 1000, threads_per_block = 256
// num_blocks = (1000 + 255) / 256 = 4
// 总线程数 = 4 × 256 = 1024

__global__ void kernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // ❌ 没有边界检查：Thread 1000-1023会越界访问！
    data[i] = data[i] * 2.0f;
    
    // ✅ 有边界检查：安全
    if (i < n) {
        data[i] = data[i] * 2.0f;
    }
}
```

---

## 8. 常见错误与调试技巧

### 8.1 错误1：忘记边界检查

```cuda
// ❌ 错误
__global__ void kernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    data[i] = 0;  // 可能越界
}

// ✅ 正确
__global__ void kernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = 0;
    }
}
```

### 8.2 错误2：Block大小不是32的倍数

```cuda
// ⚠️ 不推荐：浪费Warp资源
kernel<<<num_blocks, 100>>>(args);  // 100 = 3个Warp + 4个空闲Thread

// ✅ 推荐
kernel<<<num_blocks, 128>>>(args);  // 128 = 4个完整Warp
```

### 8.3 错误3：在条件分支中使用`__syncthreads()`

```cuda
// ❌ 错误：死锁
if (threadIdx.x < 128) {
    __syncthreads();  // 只有一半Thread执行
}

// ✅ 正确
__syncthreads();
if (threadIdx.x < 128) {
    // 条件代码
}
```

### 8.4 调试技巧：打印Thread信息

```cuda
__global__ void debug_kernel() {
    if (blockIdx.x == 0 && threadIdx.x < 5) {
        printf("Block %d, Thread %d: global_idx = %d\n",
               blockIdx.x, threadIdx.x,
               blockIdx.x * blockDim.x + threadIdx.x);
    }
}
```

---

## 9. 行业案例：FlashAttention的Block设计

FlashAttention通过精心设计的Block配置实现了3-5倍的加速。

### 9.1 标准Attention的问题

```python
# 标准Attention：O(N²)内存
Q, K, V: [batch, seq_len, dim]
S = Q @ K.T              # [batch, seq_len, seq_len] - 巨大！
P = softmax(S)           # [batch, seq_len, seq_len]
O = P @ V                # [batch, seq_len, dim]
```

**问题**：当seq_len=2048时，S和P需要2048×2048×4字节 = 16MB（每个batch）

### 9.2 FlashAttention的Tiling策略

```cuda
// 伪代码：FlashAttention的Block配置
dim3 block(128, 1);  // 每个Block处理128个query
dim3 grid(seq_len / 128, batch_size);

__global__ void flash_attention(Q, K, V, O) {
    __shared__ float shared_K[128][64];  // Tile of K
    __shared__ float shared_V[128][64];  // Tile of V
    
    // 每个Block处理128个query
    int q_start = blockIdx.x * 128;
    
    // 分块加载K和V，逐块计算
    for (int k_start = 0; k_start < seq_len; k_start += 128) {
        // 加载K和V的Tile到Shared Memory
        load_tile(shared_K, K, k_start);
        load_tile(shared_V, V, k_start);
        __syncthreads();
        
        // 计算当前Tile的Attention
        compute_attention_tile(Q, shared_K, shared_V, O, q_start, k_start);
        __syncthreads();
    }
}
```

**关键设计**：
- Block大小128：平衡Shared Memory使用和并行度
- Tiling：将大矩阵分块，每次只加载一个Tile到Shared Memory
- 在线Softmax：避免存储完整的S矩阵

---

## 总结与思考题

### 核心要点回顾

1. **三级层级**：Grid → Block → Thread，每级都有独立的索引和维度
2. **Thread**：GPU执行的最小单元，通过blockIdx和threadIdx计算全局索引
3. **Block**：协作的线程组，共享Shared Memory，可以同步
4. **Grid**：kernel的执行空间，Block独立执行，顺序不确定
5. **硬件映射**：Block → SM，Thread → Warp（32个Thread）
6. **配置原则**：Block大小256，是32的倍数，Grid足够大

### 思考题

1. **为什么Block大小要是32的倍数？** 提示：Warp的大小
2. **如果Block太大（如1024），会有什么问题？** 提示：资源限制
3. **为什么Block之间不能通信？** 提示：可扩展性设计

### 动手练习

1. 修改向量加法代码，尝试不同的Block大小（64, 128, 256, 512），对比性能
2. 实现2D矩阵转置，使用2D Grid和Block配置
3. 打印前10个Thread的blockIdx、threadIdx和全局索引

---

## 参考资料

[CUDA-PG-13.1] NVIDIA. "CUDA C++ Programming Guide v13.1 - Programming Model". https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model

[Kirk2022-PMPP] Kirk, D. B., & Hwu, W. W. (2022). "Programming Massively Parallel Processors" (4th ed.). Chapter 2: Heterogeneous Data Parallel Computing.

[Harris2022-Model] Harris, M. (2022). "CUDA Refresher: The CUDA Programming Model". NVIDIA Developer Blog.

[Dao2022-FA] Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness". NeurIPS 2022.

---

## 下期预告

**第4篇：💾 GPU内存体系：从全局内存到寄存器**

下一篇我们将深入学习：
- GPU的内存层级（Global/Shared/Register）
- 内存带宽与延迟
- Coalesced Access（合并访问）
- 如何优化内存访问模式

准备好理解GPU性能的关键了吗？让我们下期见！👋

---

*本文是《CUDA修仙之路》系列的第3篇，共38篇*  
*最后更新：2026年2月27日*
