# 🛠️ 从零搭建CUDA开发环境（Windows/Linux/WSL2）

> **系列**：CUDA修仙之路
> **难度**：⭐⭐ (1-5星)  
> **前置知识**：第1篇 - 为什么学CUDA  
> **预计阅读时间**：20分钟  
> **配套代码**：[GitHub链接](https://github.com/Mark930209/DaoOfCUDA)

---

## 开篇：工欲善其事，必先利其器

在上一篇文章中，我们看到了CUDA的强大威力。但在开始编写CUDA程序之前，我们需要搭建一个完整的开发环境。

很多初学者在这一步就遇到了困难：
- ❌ "CUDA安装失败，提示驱动版本不匹配"
- ❌ "nvcc编译报错，找不到头文件"
- ❌ "程序运行时提示no CUDA-capable device"

**本文将手把手教你**：
1. ✅ 在Windows/Linux/WSL2上安装CUDA Toolkit
2. ✅ 配置VSCode + Nsight开发环境
3. ✅ 编译运行第一个CUDA程序
4. ✅ 解决常见安装问题

**如果你没有NVIDIA GPU**，别担心！我们还会介绍：
- 🌐 Google Colab（免费GPU）
- ☁️ 云服务器（AWS/Azure/GCP）

---

## 1. 硬件与软件要求

### 1.1 硬件要求

| 组件 | 最低要求 | 推荐配置 | 说明 |
|------|---------|---------|------|
| **GPU** | NVIDIA GPU (Compute Capability ≥ 3.5) | RTX 3060/4060 (12GB) | 查看GPU型号：`nvidia-smi` |
| **显存** | 2GB | 8GB+ | 显存越大，能处理的数据越多 |
| **内存** | 8GB | 16GB+ | 编译大型项目需要更多内存 |
| **硬盘** | 10GB可用空间 | 50GB+ SSD | CUDA Toolkit约3GB |

#### 如何查看你的GPU型号？

**Windows**：
```powershell
# 方法1：任务管理器 → 性能 → GPU
# 方法2：命令行
nvidia-smi
```

**Linux**：
```bash
lspci | grep -i nvidia
nvidia-smi
```

#### 计算能力（Compute Capability）对照表

| GPU系列 | 计算能力 | 支持的CUDA版本 | 代表型号 |
|---------|---------|---------------|---------|
| Kepler | 3.5 | CUDA 5.0+ | GTX 780 |
| Maxwell | 5.0-5.2 | CUDA 6.0+ | GTX 980 |
| Pascal | 6.0-6.2 | CUDA 8.0+ | GTX 1080, P100 |
| Volta | 7.0 | CUDA 9.0+ | V100 |
| Turing | 7.5 | CUDA 10.0+ | RTX 2080 |
| Ampere | 8.0-8.6 | CUDA 11.0+ | A100, RTX 3090 |
| Hopper | 9.0 | CUDA 11.8+ | H100 |
| Blackwell | 10.0 | CUDA 13.0+ | B200, RTX 5090 |

**查看计算能力**：https://developer.nvidia.com/cuda-gpus

### 1.2 软件要求

| 软件 | 版本要求 | 说明 |
|------|---------|------|
| **操作系统** | Windows 10/11, Linux (Ubuntu 20.04+), WSL2 | macOS不支持CUDA |
| **CUDA Toolkit** | 11.8+ (推荐13.1) | 包含nvcc编译器、库、工具 |
| **GPU驱动** | 与CUDA版本匹配 | 驱动版本 ≥ CUDA要求的最低版本 |
| **C++编译器** | GCC 7+ (Linux), MSVC 2019+ (Windows) | nvcc需要宿主编译器 |
| **Python** | 3.8+ (可选) | 用于PyTorch CUDA Extension |

#### CUDA版本与驱动版本对应关系

| CUDA版本 | 最低驱动版本 (Linux) | 最低驱动版本 (Windows) |
|----------|---------------------|----------------------|
| CUDA 11.8 | 520.61.05 | 522.06 |
| CUDA 12.0 | 525.60.13 | 527.41 |
| CUDA 12.4 | 550.54.14 | 551.61 |
| CUDA 13.1 | 570.00 | 571.00 |

**重要**：驱动版本必须 ≥ CUDA要求的最低版本，但不需要完全匹配。

---

## 2. 方案选择：本地 vs 云端

### 2.1 四种方案对比

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **本地Windows** | 性能最好，无网络依赖 | 安装复杂，可能有兼容性问题 | 有NVIDIA GPU的Windows用户 |
| **本地Linux** | 安装简单，性能好 | 需要Linux基础 | 有NVIDIA GPU的Linux用户 |
| **WSL2** | Windows上使用Linux工具链 | 性能略低于原生Linux | Windows用户，想用Linux工具 |
| **Google Colab** | 免费GPU，无需安装 | 网络依赖，会话有时间限制 | 学习、实验、没有GPU |
| **云服务器** | 按需使用，GPU型号多 | 按小时收费 | 生产环境、大规模训练 |

### 2.2 推荐方案

| 你的情况 | 推荐方案 | 理由 |
|---------|---------|------|
| 有NVIDIA GPU + Windows | 本地Windows 或 WSL2 | 性能最好 |
| 有NVIDIA GPU + Linux | 本地Linux | 最简单 |
| 没有NVIDIA GPU | Google Colab | 免费且够用 |
| 需要高端GPU (A100/H100) | 云服务器 | 本地买不起 |

---

## 3. 方案一：Windows本地安装

### 3.1 安装步骤

#### Step 1: 安装Visual Studio（必需）

CUDA需要MSVC编译器作为宿主编译器。

1. 下载 **Visual Studio 2022 Community**（免费）
   - 网址：https://visualstudio.microsoft.com/downloads/

2. 安装时选择 **"使用C++的桌面开发"** 工作负载
   - 包含MSVC编译器、Windows SDK

3. 验证安装：
```powershell
# 打开"Developer Command Prompt for VS 2022"
cl
# 应该显示Microsoft C/C++ Compiler版本信息
```

#### Step 2: 更新GPU驱动

1. 下载最新驱动：https://www.nvidia.com/Download/index.aspx
2. 选择你的GPU型号，下载并安装
3. 重启电脑
4. 验证：
```powershell
nvidia-smi
# 应该显示GPU信息和驱动版本
```

#### Step 3: 安装CUDA Toolkit

1. 下载CUDA Toolkit 13.1：
   - 网址：https://developer.nvidia.com/cuda-downloads
   - 选择：Windows → x86_64 → 10/11 → exe (local)

2. 运行安装程序：
   - 选择 **"自定义安装"**
   - 勾选：
     - ✅ CUDA Toolkit
     - ✅ CUDA Samples
     - ✅ CUDA Documentation
     - ✅ Nsight Compute
     - ✅ Nsight Systems
   - 安装路径：默认 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1`

3. 验证安装：
```powershell
# 检查nvcc版本
nvcc --version
# 应该显示：Cuda compilation tools, release 13.1

# 检查环境变量
echo %CUDA_PATH%
# 应该显示：C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
```

#### Step 4: 编译测试程序

```powershell
# 进入CUDA Samples目录
cd "C:\ProgramData\NVIDIA Corporation\CUDA Samples\v13.1\1_Utilities\deviceQuery"

# 编译
nvcc deviceQuery.cpp -o deviceQuery.exe

# 运行
.\deviceQuery.exe
```

**预期输出**：
```
Device 0: "NVIDIA GeForce RTX 4090"
  CUDA Capability Major/Minor version number:    8.9
  Total amount of global memory:                 24576 MBytes
  ...
Result = PASS
```

### 3.2 常见问题

#### 问题1：nvcc找不到MSVC编译器

**错误信息**：
```
nvcc fatal : Cannot find compiler 'cl.exe' in PATH
```

**解决方案**：
```powershell
# 方法1：使用"Developer Command Prompt for VS 2022"
# 方法2：手动添加到PATH
set PATH=%PATH%;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.xx.xxxxx\bin\Hostx64\x64
```

#### 问题2：驱动版本不匹配

**错误信息**：
```
CUDA driver version is insufficient for CUDA runtime version
```

**解决方案**：
1. 更新GPU驱动到最新版本
2. 或者安装较旧版本的CUDA Toolkit

#### 问题3：找不到cuda_runtime.h

**错误信息**：
```
fatal error: cuda_runtime.h: No such file or directory
```

**解决方案**：
```powershell
# 检查CUDA_PATH环境变量
echo %CUDA_PATH%

# 如果为空，手动设置
setx CUDA_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
```

---

## 4. 方案二：Linux本地安装（推荐）

### 4.1 安装步骤（Ubuntu 22.04为例）

#### Step 1: 更新系统

```bash
sudo apt update
sudo apt upgrade -y
```

#### Step 2: 安装GCC编译器

```bash
# 安装GCC和必要的工具
sudo apt install -y build-essential

# 验证
gcc --version
# 应该显示gcc (Ubuntu 11.x.x) 或更高版本
```

#### Step 3: 安装GPU驱动

**方法1：使用Ubuntu的驱动管理器（推荐）**

```bash
# 查看推荐的驱动
ubuntu-drivers devices

# 安装推荐的驱动
sudo ubuntu-drivers autoinstall

# 或者手动安装特定版本
sudo apt install nvidia-driver-550

# 重启
sudo reboot

# 验证
nvidia-smi
```

**方法2：从NVIDIA官网下载.run文件**

```bash
# 下载驱动（以550.54.14为例）
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/550.54.14/NVIDIA-Linux-x86_64-550.54.14.run

# 禁用nouveau驱动
sudo bash -c "echo blacklist nouveau > /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
sudo bash -c "echo options nouveau modeset=0 >> /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
sudo update-initramfs -u
sudo reboot

# 安装驱动
sudo sh NVIDIA-Linux-x86_64-550.54.14.run
```

#### Step 4: 安装CUDA Toolkit

**方法1：使用APT包管理器（推荐）**

```bash
# 添加NVIDIA的APT仓库
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# 安装CUDA Toolkit 13.1
sudo apt install cuda-toolkit-13-1

# 设置环境变量
echo 'export PATH=/usr/local/cuda-13.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证
nvcc --version
```

**方法2：使用.run安装文件**

```bash
# 下载CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/13.1.0/local_installers/cuda_13.1.0_570.00_linux.run

# 运行安装程序
sudo sh cuda_13.1.0_570.00_linux.run

# 安装时选择：
# - 不安装驱动（已经安装过了）
# - 安装CUDA Toolkit
# - 安装CUDA Samples

# 设置环境变量（同上）
```

#### Step 5: 编译测试程序

```bash
# 进入CUDA Samples目录
cd /usr/local/cuda-13.1/samples/1_Utilities/deviceQuery

# 编译
make

# 运行
./deviceQuery
```

### 4.2 常见问题

#### 问题1：nouveau驱动冲突

**错误信息**：
```
NVIDIA kernel module is not loaded
```

**解决方案**：
```bash
# 禁用nouveau
sudo bash -c "echo blacklist nouveau > /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
sudo update-initramfs -u
sudo reboot
```

#### 问题2：GCC版本不兼容

**错误信息**：
```
unsupported GNU version! gcc versions later than 11 are not supported!
```

**解决方案**：
```bash
# 安装GCC 11
sudo apt install gcc-11 g++-11

# 设置为默认编译器
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100
```

---

## 5. 方案三：WSL2（Windows用户的Linux体验）

### 5.1 为什么选择WSL2？

- ✅ 在Windows上使用Linux工具链
- ✅ 性能接近原生Linux（比虚拟机快）
- ✅ 可以直接访问Windows文件系统
- ✅ NVIDIA官方支持WSL2 + CUDA

### 5.2 安装步骤

#### Step 1: 启用WSL2

```powershell
# 以管理员身份运行PowerShell

# 启用WSL
wsl --install

# 重启电脑

# 设置WSL2为默认版本
wsl --set-default-version 2

# 安装Ubuntu 22.04
wsl --install -d Ubuntu-22.04
```

#### Step 2: 在Windows上安装GPU驱动

**重要**：WSL2不需要在Linux内安装驱动，只需要Windows驱动！

1. 下载并安装最新的NVIDIA驱动（同方案一）
2. 确保驱动版本 ≥ 520.61.05

#### Step 3: 在WSL2中安装CUDA Toolkit

```bash
# 进入WSL2
wsl

# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装CUDA Toolkit（同方案二）
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-13-1

# 设置环境变量
echo 'export PATH=/usr/local/cuda-13.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证
nvidia-smi  # 应该能看到GPU信息
nvcc --version
```

#### Step 4: 测试CUDA程序

```bash
# 编译deviceQuery
cd /usr/local/cuda-13.1/samples/1_Utilities/deviceQuery
make
./deviceQuery
```

### 5.3 WSL2特有问题

#### 问题：nvidia-smi显示"Failed to initialize NVML"

**解决方案**：
1. 确保Windows驱动版本足够新
2. 重启WSL：`wsl --shutdown`，然后重新进入

---

## 6. 方案四：Google Colab（零安装，立即开始）

### 6.1 为什么选择Colab？

- ✅ **完全免费**（有使用时间限制）
- ✅ **无需安装**，浏览器即可使用
- ✅ **提供免费GPU**：T4 (16GB) 或 A100 (40GB, Colab Pro)
- ✅ **预装CUDA**，开箱即用

### 6.2 使用步骤

#### Step 1: 创建Notebook

1. 访问：https://colab.research.google.com/
2. 点击 **"新建笔记本"**
3. 点击 **"运行时" → "更改运行时类型" → "硬件加速器" → "GPU"**

#### Step 2: 验证GPU

```python
# 检查GPU
!nvidia-smi

# 检查CUDA版本
!nvcc --version
```

#### Step 3: 编写和运行CUDA代码

```python
# 创建CUDA文件
%%writefile vector_add.cu
#include <stdio.h>

__global__ void vector_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    printf("Hello from CUDA!\\n");
    return 0;
}

# 编译
!nvcc vector_add.cu -o vector_add

# 运行
!./vector_add
```

### 6.3 Colab的限制

| 限制项 | 免费版 | Colab Pro ($10/月) |
|--------|--------|-------------------|
| **GPU型号** | T4 (16GB) | T4/V100/A100 |
| **连续使用时间** | 最多12小时 | 最多24小时 |
| **空闲断开时间** | 90分钟 | 更长 |
| **并发Notebook** | 1个 | 多个 |

**适用场景**：
- ✅ 学习CUDA编程
- ✅ 运行小规模实验
- ❌ 长时间训练（会被断开）
- ❌ 生产环境

---

## 7. 配置VSCode开发环境

### 7.1 安装VSCode和插件

#### Step 1: 安装VSCode

下载：https://code.visualstudio.com/

#### Step 2: 安装必要插件

在VSCode中按 `Ctrl+Shift+X` 打开扩展市场，搜索并安装：

| 插件名称 | 功能 | 必需性 |
|---------|------|--------|
| **C/C++** (Microsoft) | C++语法高亮、智能提示 | 必需 |
| **CUDA C++** (NVIDIA) | CUDA语法高亮 | 必需 |
| **Nsight Visual Studio Code Edition** | CUDA调试 | 推荐 |
| **CMake Tools** | CMake项目管理 | 推荐 |
| **Remote - WSL** | WSL2开发 | WSL2用户必需 |

### 7.2 配置IntelliSense

创建 `.vscode/c_cpp_properties.json`：

```json
{
    "configurations": [
        {
            "name": "Linux",
            "includePath": [
                "${workspaceFolder}/**",
                "/usr/local/cuda-13.1/include"
            ],
            "defines": [],
            "compilerPath": "/usr/bin/gcc",
            "cStandard": "c17",
            "cppStandard": "c++17",
            "intelliSenseMode": "linux-gcc-x64"
        },
        {
            "name": "Win32",
            "includePath": [
                "${workspaceFolder}/**",
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include"
            ],
            "defines": [],
            "compilerPath": "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.xx.xxxxx/bin/Hostx64/x64/cl.exe",
            "cStandard": "c17",
            "cppStandard": "c++17",
            "intelliSenseMode": "windows-msvc-x64"
        }
    ],
    "version": 4
}
```

### 7.3 配置编译任务

创建 `.vscode/tasks.json`：

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Build CUDA",
            "type": "shell",
            "command": "nvcc",
            "args": [
                "-g",
                "${file}",
                "-o",
                "${fileDirname}/${fileBasenameNoExtension}"
            ],
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "problemMatcher": ["$nvcc"],
            "detail": "Compile CUDA file with nvcc"
        }
    ]
}
```

### 7.4 配置调试

创建 `.vscode/launch.json`：

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "CUDA C++: Launch",
            "type": "cuda-gdb",
            "request": "launch",
            "program": "${workspaceFolder}/${fileBasenameNoExtension}",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": false
        }
    ]
}
```

### 7.5 使用VSCode编译和运行

1. 打开 `.cu` 文件
2. 按 `Ctrl+Shift+B` 编译
3. 按 `F5` 调试运行

---

## 8. 验证安装：完整测试清单

### 8.1 基础测试

```bash
# 1. 检查GPU
nvidia-smi

# 2. 检查nvcc
nvcc --version

# 3. 检查CUDA库
ls /usr/local/cuda-13.1/lib64/  # Linux
dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\lib\x64"  # Windows
```

### 8.2 编译测试

```bash
# 编译deviceQuery
cd /usr/local/cuda-13.1/samples/1_Utilities/deviceQuery
make
./deviceQuery
```

**预期输出关键信息**：
```
Device 0: "..."
  CUDA Capability Major/Minor version number:    X.X
  Total amount of global memory:                 XXXX MBytes
  ...
Result = PASS
```

### 8.3 性能测试

```bash
# 编译bandwidthTest
cd /usr/local/cuda-13.1/samples/1_Utilities/bandwidthTest
make
./bandwidthTest
```

**预期输出**：
```
Host to Device Bandwidth, 1 Device(s)
 Transfer Size (Bytes)  Bandwidth(GB/s)
 ...
 33554432               12.5

Device to Host Bandwidth, 1 Device(s)
 Transfer Size (Bytes)  Bandwidth(GB/s)
 ...
 33554432               13.2
```

---

## 9. 常见问题汇总

### 问题1：多个CUDA版本共存

**场景**：需要同时使用CUDA 11.8和13.1

**解决方案**：
```bash
# Linux：使用软链接切换
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-13.1 /usr/local/cuda

# 或者在编译时指定
nvcc -ccbin /usr/local/cuda-11.8/bin/nvcc ...
```

### 问题2：权限问题

**错误信息**：
```
Permission denied
```

**解决方案**：
```bash
# 将用户添加到video组
sudo usermod -a -G video $USER
sudo reboot
```

### 问题3：找不到libcudart.so

**错误信息**：
```
error while loading shared libraries: libcudart.so.13.1
```

**解决方案**：
```bash
# 添加到LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:$LD_LIBRARY_PATH

# 或者永久添加
sudo bash -c "echo /usr/local/cuda-13.1/lib64 > /etc/ld.so.conf.d/cuda.conf"
sudo ldconfig
```

---

## 总结与思考题

### 核心要点回顾

1. **硬件要求**：NVIDIA GPU (Compute Capability ≥ 3.5)，驱动版本匹配CUDA版本
2. **四种方案**：本地Windows/Linux、WSL2、Google Colab，根据需求选择
3. **安装顺序**：驱动 → CUDA Toolkit → 验证 → 配置IDE
4. **常见问题**：驱动不匹配、编译器版本、环境变量

### 思考题

1. **为什么WSL2不需要在Linux内安装GPU驱动？** 提示：WSL2的架构设计
2. **如果你有多个CUDA项目需要不同版本，如何管理？** 提示：Docker容器
3. **Google Colab的GPU是共享的吗？** 提示：查看nvidia-smi的进程列表

### 下一步

恭喜你完成了开发环境搭建！下一篇我们将深入学习CUDA的核心概念：**Grid、Block、Thread的层级结构**。

---

## 参考资料

[CUDA-Install-Linux] NVIDIA. "CUDA Installation Guide for Linux". https://docs.nvidia.com/cuda/cuda-installation-guide-linux/

[CUDA-Install-Windows] NVIDIA. "CUDA Installation Guide for Windows". https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/

[WSL2-CUDA] NVIDIA. "CUDA on WSL User Guide". https://docs.nvidia.com/cuda/wsl-user-guide/

[Colab-GPU] Google. "Using GPUs in Colab". https://colab.research.google.com/notebooks/gpu.ipynb

---

## 下期预告

**第3篇：🧵 CUDA编程模型：Grid、Block、Thread的前世今生**

下一篇我们将学习：
- CUDA的三级线程层级
- 如何映射到GPU硬件
- `<<<grid, block>>>` 语法的含义
- 第一个真正的并行程序

准备好理解GPU并行的本质了吗？让我们下期见！👋

---

*本文是《CUDA从入门到精通》系列的第2篇，共38篇*  
*最后更新：2026年2月26日*
