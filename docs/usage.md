# 使用指南

## 快速开始

### 1. 准备 VTK 数据

渲染器需要以下输入数据：
- 一个.vtk.series序列文件
- 一组.vtk文件，或一组.cache缓存文件
- config.json配置文件

### 2. 配置项目

编辑 `files/config.json` 文件，设置以下关键参数：

```json
{
  "series-path": "../files/",
  "series-name": "particle_mesh-short.vtk.series",
  "cache-path": "../cache/"
}
```

### 3. 运行程序

#### Windows
```powershell
cd bin
.\RendererOptiX.exe
```

#### Linux
```bash
cd bin
./RendererOptiX
```

## 配置文件说明

### 基本配置

#### 文件路径配置

- `series-path`：VTK 序列文件所在目录（相对或绝对路径）
- `series-name`：序列文件名（包含 `.series` 后缀）
- `cache-path`：缓存文件存放目录

#### 缓存配置

- `cache`：`true` 表示生成缓存文件后退出，`false` 表示正常渲染
- `cache-process-thread-count`：读写缓存时使用的 CPU 线程数（推荐 4-8，根据实际CPU型号和系统内存大小进行设置，越高的并发线程数，对系统内存的要求越高。每个线程约需要1G内存）

#### 调试模式

- `debug-mode`：`true` 启用 OptiX 和图形 API 的调试模式（性能较低）

### 材质配置

#### 粗糙材质 (Rough)

```json
"roughs": [
  {"albedo": [0.65, 0.05, 0.05]},
  {"albedo": [0.73, 0.73, 0.73]},
  {"albedo": [0.12, 0.45, 0.15]},
  {"albedo": [0.70, 0.60, 0.50]}
]
```

- `albedo`：反照率颜色（RGB，范围 0.0-1.0）

#### 金属材质 (Metal)

```json
"metals": [
  {"albedo": [0.8, 0.85, 0.88], "fuzz": 0.0}  // 镜面金属
]
```

- `albedo`：金属颜色（RGB，范围 0.0-1.0）
- `fuzz`：表面粗糙度（范围 0.0-1.0，0.0 为完全镜面，值越大越粗糙）

### 几何体配置

#### 球体 (Spheres)

```json
"spheres": [
  {
    "center": [0.0, 0.0, 0.0],
    "radius": 1000.0,
    "mat-type": "ROUGH",
    "mat-index": 3,
    "shift": [0.0, 0.0, -1000.5],
    "rotate": [0.0, 0.0, 0.0],
    "scale": [1.0, 1.0, 1.0]
  }
]
```

- `center`：球心位置
- `radius`：球体半径
- `mat-type`：材质类型（`"ROUGH"` 或 `"METAL"`）
- `mat-index`：材质索引（对应 `roughs` 或 `metals` 数组索引）
- `shift`：位移向量
- `rotate`：旋转角度（度）
- `scale`：缩放因子

#### 三角形 (Triangles)

```json
"triangles": [
  {
    "vertices": [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]],
    "normals": [[nx1, ny1, nz1], [nx2, ny2, nz2], [nx3, ny3, nz3]],
    "mat-type": "ROUGH",
    "mat-index": 0
  }
]
```

### 渲染循环配置

```json
"loop-data": {
  "api": "VK",
  "window-width": 1200,
  "window-height": 800,
  "fps": 60,
  "camera-center": [5.0, 0.0, 0.0],
  "camera-target": [0.0, 0.0, 0.0],
  "up-direction": [0.0, 0.0, 1.0],
  "camera-pitch-limit-degree": 85.0,
  "camera-speed-stride": 0.002,
  "camera-initial-speed-ratio": 10,
  "mouse-sensitivity": 0.002,
  "render-speed-ratio": 4,
  "particle-shift": [0.0, 0.0, 0.0],
  "particle-scale": [1.0, 1.0, 1.0]
}
```

#### 图形 API 选择

- `api`：图形 API 类型
  - `"OGL"`：OpenGL
  - `"VK"`：Vulkan
  - `"D3D11"`：Direct3D 11（仅 Windows）
  - `"D3D12"`：Direct3D 12（仅 Windows）

#### 窗口设置

- `window-width`：窗口宽度（像素）
- `window-height`：窗口高度（像素）
- `fps`：目标帧率

#### 相机设置

- `camera-center`：初始相机位置
- `camera-target`：初始相机目标点
- `up-direction`：相机上方向量（无需单位化）
- `camera-pitch-limit-degree`：俯仰角限制（度，应小于 90）
- `camera-speed-stride`：滚轮调节速度的变化量
- `camera-initial-speed-ratio`：初始速度相对于 `camera-speed-stride` 的倍数
- `mouse-sensitivity`：鼠标灵敏度

#### 粒子设置

- `render-speed-ratio`：粒子运动速度倍率（越大越慢，1 为原速）
- `particle-shift`：所有粒子的整体位移
- `particle-scale`：所有粒子的整体缩放

## 交互控制

### 键盘控制

- **W**：向前移动
- **S**：向后移动
- **A**：向左移动
- **D**：向右移动
- **Space**：向上移动
- **Left Shift**：向下移动
- **Tab**：按下时禁用降噪
- **ESC**：退出程序

### 鼠标控制

- **鼠标移动**：旋转相机视角
- **鼠标滚轮**：调节相机移动速度。向上滚动提升速度

## 使用示例

### 示例 1：基本渲染

1. 准备 VTK 数据文件
2. 编辑 `config.json`，设置正确的文件路径
3. 运行程序

```json
{
  "series-path": "../files/",
  "series-name": "particle_mesh.vtk.series",
  "cache-path": "../cache/",
  "cache": false,
  "debug-mode": false,
  "loop-data": {
    "api": "VK",
    "window-width": 1920,
    "window-height": 1080,
    "fps": 60
  }
}
```

### 示例 2：生成缓存

如果首次加载 VTK 数据较慢，可以先生成缓存：

1. 设置 `"cache": true`
2. 运行程序，程序会在生成缓存后自动退出
3. 将 `"cache": false` 改回，正常渲染

#### 什么时候需要重新生成缓存
1. series文件引用的任何一个VTK文件被修改
2. 更换运行平台，如从Windows换到Linux
3. 因缓存文件损坏导致渲染错误

重新生成缓存时，只需要将cache参数改为true并运行程序即可，原有缓存文件会被自动删除。

### 示例 3：添加环境球体

添加一个大的环境球体作为背景：

```json
"spheres": [
  {
    "center": [0.0, 0.0, 0.0],
    "radius": 1000.0,
    "mat-type": "ROUGH",
    "mat-index": 3,
    "shift": [0.0, 0.0, -1000.5],
    "rotate": [0.0, 0.0, 0.0],
    "scale": [1.0, 1.0, 1.0]
  }
]
```

### 示例 4：多材质场景

```json
"roughs": [
  {"albedo": [0.8, 0.1, 0.1]},  // 红色
  {"albedo": [0.1, 0.8, 0.1]},  // 绿色
  {"albedo": [0.1, 0.1, 0.8]}   // 蓝色
],
"metals": [
  {"albedo": [0.9, 0.9, 0.9], "fuzz": 0.0},  // 镜面
  {"albedo": [0.7, 0.6, 0.5], "fuzz": 0.3}   // 粗糙金属
]
```

## 性能建议

1. **使用缓存**：首次加载后生成缓存文件，后续加载会更快
2. **调整线程数**：根据 CPU 核心数设置 `cache-process-thread-count`
3. **选择合适的 API**：Vulkan 通常性能最好，OpenGL 兼容性最好
4. **降低分辨率**：如果帧率不足，可以降低窗口分辨率
5. **关闭调试模式**：生产环境确保 `debug-mode` 为 `false`

## 高级用法

### 自定义实例更新

在 `Main.cu` 中，可以自定义 `updateInstancesTransforms` 函数来实现动态几何体变换：

```cpp
static void updateInstancesTransforms(
    OptixInstance * pin_instances, size_t instanceCount, unsigned long long frameCount)
{
    // 根据 frameCount 更新变换矩阵
    // 例如：旋转动画
    float angle = frameCount * 0.01f;
    // ... 计算变换矩阵并更新 pin_instances
    // 只需要将新的变换矩阵拷贝到pin_instances[i].transform即可
    // 此函数中只修改额外几何体的实例变换矩阵，所有额外几何体实例按照 球体 -> 三角形 的顺序从实例数组收个元素开始向后存放，instanceCount为当前文件包含所有粒子实例的实例总数
    // frameCount为当前帧数
}
```
修改完成后，需要重新编译项目，无需重新生成缓存文件
