# RendererOptiX

一个基于 NVIDIA OptiX 实现的 VTK 渲染器，支持实时渲染特定格式的 VTK 粒子数据，并支持使用OpenGL、Vulkan、Direct3D（Windows）呈现画面。

## 项目简介

RendererOptiX 是一个高性能实时渲染器，使用 NVIDIA OptiX 9.0 进行光线追踪渲染。该项目能够读取和渲染特定 VTK 格式的粒子数据，并生成缓存文件。支持粗糙材质和金属材质类型，提供键鼠交互式控制。

## 功能特性

- **高性能光线追踪**：基于 NVIDIA OptiX 9.0 实现
- **VTK 数据支持**：支持读取和渲染特定格式的 VTK 粒子序列文件
- **多种材质**：支持粗糙材质（Rough）和金属材质（Metal）
- **多图形 API**：支持 OpenGL、Vulkan、Direct3D11、Direct3D12
- **交互式控制**：支持鼠标和键盘控制相机
- **缓存系统**：支持 VTK 数据缓存以加速加载
- **可配置渲染**：通过 JSON 配置文件灵活配置渲染参数

## 文档索引

- [项目介绍](docs/project-introduction.md) - 项目概述、架构设计和核心概念
- [使用指南](docs/usage.md) - 快速开始、配置说明和使用示例
- [配置参考](docs/configuration.md) - 配置文件格式和参数说明
- [构建指南](docs/build-guide.md) - 编译环境要求、构建步骤和依赖配置
- [技术细节](docs/technical-details.md) - 技术架构、实现原理和性能优化

## 快速开始

**注意：在满足重新生成的要求时，需要重新编译着色器PTX文件或重新生成缓存，否则渲染器无法正常工作，具体要求请参见文档对应部分。**

### Windows

### Ubuntu
