# 构建指南

## 系统要求

### 硬件要求

- **GPU**：NVIDIA GPU，支持 CUDA 和 OptiX
  - 推荐：RTX 20 系列或更高
  - 最低：支持 CUDA Compute Capability 7.0+
- **内存**：至少 8GB RAM（推荐 16GB 或更多）
- **显存**：至少 8GB（推荐 12GB 或更多，取决于VTK粒子规模）

### 软件要求

#### Windows

- **操作系统**：Windows 10 (1903+) 或 Windows 11
- **编译器**：Visual Studio 2019 或更高版本（支持 C++20）
- **CMake**：4.0 或更高版本
- **CUDA Toolkit**：与 OptiX 9.0.0 兼容的版本（推荐 11.0+）
- **NVIDIA OptiX SDK**：9.0.0
- **Vulkan SDK**: 1.1及以上

#### Linux

- **操作系统**：Ubuntu 20.04+ 或类似发行版
- **编译器**：GCC 10+ 或 Clang 12+（支持 C++20）
- **CMake**：4.0 或更高版本
- **CUDA Toolkit**：与 OptiX 9.0.0 兼容的版本
- **NVIDIA OptiX SDK**：9.0.0

## 依赖库安装

### NVIDIA OptiX SDK

#### Windows

1. 从 NVIDIA 官网下载 OptiX SDK 9.0.0
2. 安装到默认位置：`C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.0.0`
3. 或者修改 `CMakeLists.txt` 中的 `OPTIX_ROOT` 路径

#### Linux

1. 下载 OptiX SDK 9.0.0
2. 解压到用户目录，例如：`~/Desktop/CUDA/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64`
3. 确保路径与 `CMakeLists.txt` 中的 `OPTIX_ROOT` 一致

### CUDA Toolkit

1. 从 NVIDIA 官网下载并安装 CUDA Toolkit
2. 确保 `nvcc` 在系统 PATH 中
3. Windows 会自动检测，Linux 需要设置环境变量或修改 `CMakeLists.txt`

### VTK

项目已包含预编译的 VTK 9.5.2 库在 `lib/VTK-9.5.2/` 目录。

如果需要使用系统安装的 VTK：
1. 安装 VTK 9.5.2
2. 修改 `CMakeLists.txt`，使用 `find_package(VTK)` 替代硬编码路径

### SDL2

#### Windows

项目已包含预编译的 SDL2 库在 `lib/` 目录。

#### Linux

使用系统包管理器安装：
```bash
sudo apt install libsdl2-dev libsdl2-ttf-dev
```

### Vulkan SDK

#### Windows
1. 从 Vulkan 官网下载并安装 Vulkan SDK
2. 确保环境变量正确设置

#### Linux
```bash
sudo apt-get install vulkan-sdk
```

## 构建步骤

### Windows

#### 使用 Visual Studio

1. **打开项目**
   ```powershell
   cd RendererOptiX
   mkdir build
   cd build
   ```

2. **配置 CMake**
   ```powershell
   cmake .. -G "Visual Studio 17 2022" -A x64
   ```
   或使用其他生成器：
   ```powershell
   cmake .. -G "Visual Studio 16 2019" -A x64
   ```

3. **构建项目**
   ```powershell
   cmake --build . --config Release
   ```
   或打开生成的 `.sln` 文件，在 Visual Studio 中构建。

4. **运行**
   可执行文件位于 `bin/RendererOptiX`

### Linux

1. **创建构建目录**
   ```bash
   cd RendererOptiX
   mkdir build
   cd build
   ```

2. **配置 CMake**
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release
   ```

3. **构建项目**
   ```bash
   cmake --build . -j$(nproc)
   ```

4. **运行**
   ```bash
   ./bin/RendererOptiX
   ```

## CMake 配置选项

### 自定义 OptiX 路径

如果 OptiX 不在默认位置，修改 `CMakeLists.txt`：

```cmake
# Windows
set(OPTIX_ROOT "C:/Your/Path/To/OptiX SDK 9.0.0")

# Linux
set(OPTIX_ROOT "~/Your/Path/To/OptiX-SDK")
```

### 自定义 CUDA 路径

#### Windows
CUDA 通常会自动检测。

#### Linux
```cmake
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
```

### 构建类型

```bash
# Debug 构建
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Release 构建（推荐）
cmake .. -DCMAKE_BUILD_TYPE=Release

# RelWithDebInfo
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

## 开发环境设置

### Visual Studio Code

1. 安装扩展：
   - C/C++
   - CMake Tools
   - CUDA

2. 配置 `.vscode/settings.json`：
   ```json
   {
     "cmake.configureSettings": {
       "CMAKE_BUILD_TYPE": "Release"
     }
   }
   ```

### CLion

1. 打开项目
2. 配置 CMake：
   - Build type: Release
   - CMake options: 根据需要添加
3. 构建和运行

## 下一步

构建成功后，请参考：
- [使用指南](usage.md) - 学习如何使用程序
- [配置参考](configuration.md) - 了解配置选项
- [技术细节](technical-details.md) - 深入了解实现

