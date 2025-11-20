# RendererOptiX

一个基于 NVIDIA OptiX 实现的 VTK 渲染器，支持实时渲染 VTK 粒子数据，并提供多种图形 API 支持。

## 项目简介

RendererOptiX 是一个高性能的实时渲染器，使用 NVIDIA OptiX 9.0 进行光线追踪渲染。该项目能够读取和渲染 VTK 格式的粒子数据，支持多种材质类型（粗糙材质和金属材质），并提供交互式相机控制。

## 功能特性

- **高性能光线追踪**：基于 NVIDIA OptiX 9.0 实现
- **VTK 数据支持**：支持读取和渲染 VTK 粒子序列文件
- **多种材质**：支持粗糙材质（Rough）和金属材质（Metal）
- **多图形 API**：支持 OpenGL、Vulkan、Direct3D11、Direct3D12
- **交互式控制**：支持鼠标和键盘控制相机
- **缓存系统**：支持 VTK 数据缓存以加速加载
- **可配置渲染**：通过 JSON 配置文件灵活配置渲染参数

## 文档索引

### 主要文档

- [项目介绍](docs/project-introduction.md) - 项目概述、架构设计和核心概念
- [使用指南](docs/usage.md) - 快速开始、配置说明和使用示例
- [技术细节](docs/technical-details.md) - 技术架构、实现原理和性能优化
- [构建指南](docs/build-guide.md) - 编译环境要求、构建步骤和依赖配置
- [配置参考](docs/configuration.md) - 配置文件格式和参数说明

### 🔍 快速导航

- **新用户**：建议从 [项目介绍](docs/project-introduction.md) 和 [使用指南](docs/usage.md) 开始
- **开发者**：查看 [技术细节](docs/technical-details.md) 了解实现原理
- **构建问题**：参考 [构建指南](docs/build-guide.md) 解决编译问题

## 系统要求

### 硬件要求
- NVIDIA GPU（支持 CUDA 和 OptiX，拥有RT Core，20系Turing架构及以上）
- 至少 8GB 显存（推荐 12GB 或更多）

### 软件要求
- Windows 10/11 或 Linux
- CUDA Toolkit（与 OptiX 版本兼容）
- NVIDIA OptiX SDK 9.0.0
- CMake 4.0 或更高版本
- C++20 兼容的编译器

### 依赖库
- VTK 9.5.2
- SDL2 2.32.10
- SDL2_ttf 2.24.0
- Vulkan SDK（如果使用 Vulkan）
- DirectX SDK（Windows，如果使用 D3D11/D3D12）

## 快速开始

1. **克隆仓库**
   ```bash
   git clone https://github.com/3169651074/RendererOptiX.git
   cd RendererOptiX
   ```

2. **配置项目**
   编辑 `files/config.json` 文件，设置 VTK 文件路径和渲染参数

3. **构建项目**
   ```bash
   mkdir build
   cd build
   cmake ..
   cmake --build .
   ```

4. **运行程序**
   ```bash
   ./bin/RendererOptiX
   ```

详细的使用说明请参考 [使用指南](docs/usage.md)。

## 项目结构

```
RendererOptiX/
├── bin/              # 可执行文件输出目录
├── cache/            # VTK 缓存文件目录
├── docs/             # 文档目录
├── files/            # 配置文件和示例数据
├── include/          # 头文件
├── lib/              # 第三方库
├── shader/           # OptiX 着色器代码
└── src/              # 源文件
```
