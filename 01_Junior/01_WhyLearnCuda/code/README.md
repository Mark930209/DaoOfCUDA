# 第1篇配套代码：向量加法

## 文件说明

- `vector_add.cu` - CUDA向量加法完整实现，包含CPU/GPU性能对比
- `vector_add_simple.cu` - 简化版本，适合初学者理解基本流程

## 编译与运行

### 本地环境（需要安装CUDA Toolkit）

```bash
# 编译
nvcc vector_add.cu -o vector_add

# 运行
./vector_add
```

### Google Colab（推荐）

在Colab中运行以下代码：

```python
# 1. 检查GPU
!nvidia-smi

# 2. 创建CUDA文件
%%writefile vector_add.cu
[粘贴vector_add.cu的内容]

# 3. 编译
!nvcc vector_add.cu -o vector_add

# 4. 运行
!./vector_add
```

## 预期输出

```
=== CUDA向量加法性能对比 ===

📊 问题规模：10000000 个元素 (38.15 MB)

🔧 初始化数据...

⏱️  CPU计算中...
✅ CPU时间: 18.45 ms

⏱️  GPU计算中...
   Grid配置: 39063 blocks × 256 threads = 10000128 threads
✅ GPU总时间: 2.34 ms
   ├─ 数据拷贝到GPU: 1.23 ms
   ├─ Kernel执行: 0.45 ms
   └─ 结果拷贝回CPU: 0.66 ms

🔍 验证结果...
✅ 结果正确！CPU和GPU计算结果一致

📈 性能对比：
   CPU时间:        18.45 ms
   GPU总时间:      2.34 ms (加速比: 7.9x)
   GPU纯计算时间:  0.45 ms (加速比: 41.0x)

💾 内存吞吐量：
   CPU: 4.13 GB/s
   GPU: 169.33 GB/s

✨ 程序执行完成！
```

## 代码解析

### 1. CUDA Kernel定义

```cuda
__global__ void vector_add_cuda(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

- `__global__`: 表示这是一个在GPU上运行的kernel函数
- `blockIdx.x`: 当前block的索引
- `blockDim.x`: 每个block的线程数
- `threadIdx.x`: 当前线程在block中的索引

### 2. 内存管理

```cuda
// 分配GPU内存
cudaMalloc(&d_a, bytes);

// 拷贝数据到GPU
cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

// 拷贝结果回CPU
cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost);

// 释放GPU内存
cudaFree(d_a);
```

### 3. Kernel启动

```cuda
int threads_per_block = 256;
int blocks = (n + threads_per_block - 1) / threads_per_block;
vector_add_cuda<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
```

- `<<<blocks, threads_per_block>>>`: CUDA的kernel启动语法
- `blocks`: Grid中的block数量
- `threads_per_block`: 每个block的线程数

## 性能分析

### 为什么GPU更快？

1. **并行度**：GPU有10,000+个核心同时工作，CPU只有几个核心
2. **内存带宽**：GPU内存带宽可达1-2 TB/s，远超CPU的50-100 GB/s

### 为什么GPU总时间没有快那么多？

因为包含了数据传输开销：
- CPU ↔ GPU的数据传输通过PCIe，带宽只有16-32 GB/s
- 对于简单计算，数据传输时间可能超过计算时间

### 优化建议

1. **减少数据传输**：尽量在GPU上完成所有计算
2. **增加计算复杂度**：让计算时间远大于传输时间
3. **使用异步传输**：Overlap数据传输和计算（后续文章会讲）

## 练习题

1. **修改数据量**：将`n`改为100、1000、100000，观察性能变化
2. **修改block大小**：将`threads_per_block`改为128、512，观察性能变化
3. **添加更多操作**：将`c[i] = a[i] + b[i]`改为`c[i] = a[i] * b[i] + a[i] / b[i]`

## 常见问题

### Q1: 编译时报错 "nvcc: command not found"

**A**: 需要先安装CUDA Toolkit，参考第2篇文章的环境搭建教程。

### Q2: 运行时报错 "no CUDA-capable device is detected"

**A**: 你的机器没有NVIDIA GPU，建议使用Google Colab。

### Q3: 为什么我的GPU加速比没有这么高？

**A**: 可能原因：
- GPU型号较老（计算能力较低）
- CPU性能较强（多核并行）
- 数据量太小（GPU优势不明显）

## 下一步

学习第2篇文章，了解如何搭建完整的CUDA开发环境！
