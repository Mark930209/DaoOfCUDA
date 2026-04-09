# 配套代码说明

本目录包含《CUDA 性能分析实战：Nsight Systems 与 Nsight Compute 入门》一文的配套代码。

## 文件说明

| 文件 | 用途 | 配合工具 |
|---|---|---|
| `profile_demo.cu` | 系统级分析：baseline + 反面教材 | Nsight Systems |
| `transpose_demo.cu` | kernel 级分析：三版矩阵转置 | Nsight Compute |

### `profile_demo.cu`

包含两个阶段，用来在 Nsight Systems 的 timeline 中制造对比鲜明的场景：

1. **SAXPY 基线**：标准 SAXPY kernel，20 次取均值，含预热步骤；
2. **反面教材**：200 次极小 kernel + 每次 `cudaDeviceSynchronize()`，制造 GPU 空闲碎片。

### `transpose_demo.cu`

包含三个矩阵转置 kernel，体现逐步优化的过程：

| kernel | 特征 |
|---|---|
| `transpose_naive` | 全局写入非合并（non-coalesced） |
| `transpose_shared` | 引入 Shared Memory，但有 Bank Conflict |
| `transpose_optimized` | `tile[32][33]` padding 消除 Bank Conflict |

每个 kernel 运行后输出平均耗时、有效带宽（GB/s）和 speedup。

## 编译

**Linux / macOS**：

```bash
nvcc -O3 -lineinfo -o profile_demo profile_demo.cu
nvcc -O3 -lineinfo -o transpose_demo transpose_demo.cu
```

**Windows**（需额外加 `/utf-8` 避免 C4819 警告）：

```bash
nvcc -O3 -lineinfo -Xcompiler /utf-8 -o profile_demo.exe profile_demo.cu
nvcc -O3 -lineinfo -Xcompiler /utf-8 -o transpose_demo.exe transpose_demo.cu
```

编译选项说明：

| 选项 | 作用 |
|---|---|
| `-O3` | 最高级别编译优化，贴近实际发布版本的性能表现 |
| `-lineinfo` | 嵌入源码行号信息，让 Nsight Compute 能将指标关联到源码行 |
| `-Xcompiler /utf-8` | （仅 Windows）让 MSVC 以 UTF-8 解析源文件，消除 C4819 警告 |

> `-lineinfo` 基本不影响运行性能，分析阶段建议始终加上。

## 运行

```bash
# baseline + timeline 示例
./profile_demo

# 矩阵转置示例
./transpose_demo
```

## 分析流程

文章的核心思路：**先 Nsight Systems 看全局，再 Nsight Compute 挖热点**。

### 第一步：Nsight Systems 分析 `profile_demo`

```bash
nsys profile -o profile_demo_report ./profile_demo
```

只关心 CUDA 活动、减小报告体积：

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none -o profile_demo_report ./profile_demo
```

重点观察：

- 阶段 1 中 SAXPY 连续执行，GPU 利用率高；
- 阶段 2 中 tiny_kernel 碎片化严重，GPU 大量空闲。

### 第二步：Nsight Compute 分析 `transpose_demo`

分析全部 kernel：

```bash
ncu -o transpose_report ./transpose_demo
```

按 kernel 过滤分析：

```bash
ncu --kernel-name transpose_naive     -o naive_report     ./transpose_demo
ncu --kernel-name transpose_shared    -o shared_report    ./transpose_demo
ncu --kernel-name transpose_optimized -o optimized_report ./transpose_demo
```

采集完整指标（含 Roofline，耗时更长）：

```bash
ncu --set full -o transpose_full_report ./transpose_demo
```

重点关注三个版本之间 Global Store Throughput、Bank Conflict、有效带宽的变化。
