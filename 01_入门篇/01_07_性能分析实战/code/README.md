# 配套代码说明

本目录用于配合《性能分析实战：Nsight Systems与Nsight Compute》一文。

## 文件说明

### `profile_demo.cu`

用于演示系统级 profiling 场景，重点是：

- 如何建立一个简单、可重复的 baseline；
- 为什么预热重要；
- 为什么“很多小 kernel + 每次同步”会让 timeline 很难看；
- 如何为 `Nsight Systems` 提供可视化素材。

### `transpose_demo.cu`

用于演示 kernel 级 profiling 场景，重点是：

- `transpose_naive`
- `transpose_shared`
- `transpose_optimized`

适合配合 `Nsight Compute` 观察：

- `Summary`
- `Speed Of Light`
- `Occupancy`
- `Memory Workload Analysis`
- `Roofline`
- 三个版本之间的性能差异

## 编译方法

下面以 Windows + `nvcc` 为例。

### 编译 `profile_demo.cu`

```bash
nvcc -O3 -lineinfo -o profile_demo.exe profile_demo.cu
```

### 编译 `transpose_demo.cu`

```bash
nvcc -O3 -lineinfo -o transpose_demo.exe transpose_demo.cu
```

> `-O3` 用于尽量贴近实际优化构建。  
> `-lineinfo` 用于让 `Nsight Compute` 的源码关联视图更完整。

## 运行方法

### 运行 baseline / timeline 示例

```bash
profile_demo.exe
```

### 运行矩阵转置示例

```bash
transpose_demo.exe
```

程序会输出：

- 平均执行时间
- 简单校验结果
- 有效带宽（矩阵转置示例）
- speedup（矩阵转置示例）

## 建议的分析顺序

### 1. `Nsight Systems`

先分析：

```bash
nsys profile -o profile_demo_report ./profile_demo.exe
```

优先观察：

- GPU 是否有明显空闲；
- 小 kernel 是否过于碎片化；
- 同步是否过多；
- CPU 与 GPU 是否缺少 overlap。

### 2. `Nsight Compute`

再分析：

```bash
ncu -o transpose_demo_report ./transpose_demo.exe
```

如果你希望更聚焦某一个 kernel，可以进一步使用过滤：

```bash
ncu --kernel-name transpose_naive -o transpose_naive_report ./transpose_demo.exe
```

类似地，也可以分析：

```bash
ncu --kernel-name transpose_shared -o transpose_shared_report ./transpose_demo.exe
ncu --kernel-name transpose_optimized -o transpose_optimized_report ./transpose_demo.exe
```
