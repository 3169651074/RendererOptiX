# 配置参考

本文档详细说明 `config.json` 配置文件的所有参数。

## 配置文件位置

默认配置文件路径：`files/config.json`

可以在 `include/Util/ProgramArgumentParser.cuh` 中修改 `CONFIG_FILE_PATH` 常量来更改路径。

## 配置结构

```json
{
  "series-path": "...",
  "series-name": "...",
  "cache-path": "...",
  "cache": false,
  "debug-mode": false,
  "cache-process-thread-count": 8,
  "roughs": [...],
  "metals": [...],
  "spheres": [...],
  "triangles": [...],
  "loop-data": {...}
}
```

## 文件路径配置

### series-path

**类型**：`string`  
**说明**：VTK 序列文件所在目录  
**示例**：
```json
"series-path": "../files/"
```
**注意**：
- 可以是相对路径（相对于可执行文件）或绝对路径
- 路径分隔符使用 `/`（Windows 和 Linux 都支持）

### series-name

**类型**：`string`  
**说明**：VTK 序列文件名，包含 `.series` 后缀  
**示例**：
```json
"series-name": "particle_mesh-short.vtk.series"
```

### cache-path

**类型**：`string`  
**说明**：缓存文件存放目录  
**示例**：
```json
"cache-path": "../cache/"
```

## 缓存配置

### cache

**类型**：`boolean`  
**说明**：是否生成缓存文件并退出程序  
**默认值**：`false`  
**用法**：
- `true`：程序启动后读取 VTK 文件，生成缓存文件，然后退出
- `false`：正常渲染模式

**示例**：
```json
"cache": false
```

### cache-process-thread-count

**类型**：`integer`  
**说明**：读写缓存文件时使用的 CPU 线程数  
**默认值**：`8`  
**推荐值**：4-8（根据 CPU 核心数调整）  
**示例**：
```json
"cache-process-thread-count": 8
```

## 调试配置

### debug-mode

**类型**：`boolean`  
**说明**：是否启用 OptiX 和图形 API 的调试模式  
**默认值**：`false`  
**注意**：
- 启用后会显著降低性能
- 仅用于调试和开发
- 生产环境应设为 `false`

**示例**：
```json
"debug-mode": false
```

## 材质配置

### roughs

**类型**：`array of objects`  
**说明**：粗糙材质数组  
**结构**：
```json
"roughs": [
  {
    "albedo": [r, g, b]
  }
]
```

#### albedo

**类型**：`array of float`（3 个元素）  
**说明**：反照率颜色（RGB）  
**范围**：`[0.0, 1.0]`  
**示例**：
```json
"roughs": [
  {"albedo": [0.65, 0.05, 0.05]},  // 红色
  {"albedo": [0.73, 0.73, 0.73]},  // 灰色
  {"albedo": [0.12, 0.45, 0.15]},  // 绿色
  {"albedo": [0.70, 0.60, 0.50]}   // 米色
]
```

### metals

**类型**：`array of objects`  
**说明**：金属材质数组  
**结构**：
```json
"metals": [
  {
    "albedo": [r, g, b],
    "fuzz": float
  }
]
```

#### albedo

**类型**：`array of float`（3 个元素）  
**说明**：金属颜色（RGB）  
**范围**：`[0.0, 1.0]`  
**示例**：
```json
{"albedo": [0.8, 0.85, 0.88]}
```

#### fuzz

**类型**：`float`  
**说明**：表面粗糙度  
**范围**：`[0.0, 1.0]`  
**说明**：
- `0.0`：完全镜面反射
- 值越大，表面越粗糙
**示例**：
```json
{"albedo": [0.8, 0.85, 0.88], "fuzz": 0.0}  // 镜面
{"albedo": [0.7, 0.6, 0.5], "fuzz": 0.3}    // 粗糙金属
```

## 几何体配置

### spheres

**类型**：`array of objects`  
**说明**：球体数组  
**结构**：
```json
"spheres": [
  {
    "center": [x, y, z],
    "radius": float,
    "mat-type": "ROUGH" | "METAL",
    "mat-index": integer,
    "shift": [x, y, z],
    "rotate": [x, y, z],
    "scale": [x, y, z]
  }
]
```

#### center

**类型**：`array of float`（3 个元素）  
**说明**：球心位置  
**示例**：
```json
"center": [0.0, 0.0, 0.0]
```

#### radius

**类型**：`float`  
**说明**：球体半径  
**示例**：
```json
"radius": 1000.0
```

#### mat-type

**类型**：`string`  
**说明**：材质类型  
**可选值**：`"ROUGH"` 或 `"METAL"`  
**示例**：
```json
"mat-type": "ROUGH"
```

#### mat-index

**类型**：`integer`  
**说明**：材质索引（对应 `roughs` 或 `metals` 数组索引）  
**注意**：索引从 0 开始  
**示例**：
```json
"mat-index": 3
```

#### shift

**类型**：`array of float`（3 个元素）  
**说明**：位移向量  
**示例**：
```json
"shift": [0.0, 0.0, -1000.5]
```

#### rotate

**类型**：`array of float`（3 个元素）  
**说明**：旋转角度（度）  
**注意**：当前实现可能未完全支持，建议使用 `[0.0, 0.0, 0.0]`  
**示例**：
```json
"rotate": [0.0, 0.0, 0.0]
```

#### scale

**类型**：`array of float`（3 个元素）  
**说明**：缩放因子  
**示例**：
```json
"scale": [1.0, 1.0, 1.0]
```

### triangles

**类型**：`array of objects`  
**说明**：三角形数组（当前可能未完全实现）  
**结构**：
```json
"triangles": [
  {
    "vertices": [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]],
    "normals": [[nx1, ny1, nz1], [nx2, ny2, nz2], [nx3, ny3, nz3]],
    "mat-type": "ROUGH" | "METAL",
    "mat-index": integer
  }
]
```

## 渲染循环配置 (loop-data)

### api

**类型**：`string`  
**说明**：图形 API 类型  
**可选值**：
- `"OGL"`：OpenGL
- `"VK"`：Vulkan
- `"D3D11"`：Direct3D 11（仅 Windows）
- `"D3D12"`：Direct3D 12（仅 Windows）

**示例**：
```json
"api": "VK"
```

### window-width

**类型**：`integer`  
**说明**：窗口宽度（像素）  
**示例**：
```json
"window-width": 1200
```

### window-height

**类型**：`integer`  
**说明**：窗口高度（像素）  
**示例**：
```json
"window-height": 800
```

### fps

**类型**：`integer`  
**说明**：目标帧率  
**示例**：
```json
"fps": 60
```

### camera-center

**类型**：`array of float`（3 个元素）  
**说明**：初始相机位置  
**示例**：
```json
"camera-center": [5.0, 0.0, 0.0]
```

### camera-target

**类型**：`array of float`（3 个元素）  
**说明**：初始相机目标点（看向的位置）  
**示例**：
```json
"camera-target": [0.0, 0.0, 0.0]
```

### up-direction

**类型**：`array of float`（3 个元素）  
**说明**：相机上方向量（无需单位化）  
**示例**：
```json
"up-direction": [0.0, 0.0, 1.0]
```

### camera-pitch-limit-degree

**类型**：`float`  
**说明**：相机俯仰角限制（度）  
**范围**：应小于 90.0  
**默认值**：`85.0`  
**示例**：
```json
"camera-pitch-limit-degree": 85.0
```

### camera-speed-stride

**类型**：`float`  
**说明**：滚轮调节相机速度的变化量  
**默认值**：`0.002`  
**示例**：
```json
"camera-speed-stride": 0.002
```

### camera-initial-speed-ratio

**类型**：`integer`  
**说明**：相机初始速度相对于 `camera-speed-stride` 的倍数  
**默认值**：`10`  
**示例**：
```json
"camera-initial-speed-ratio": 10
```

### mouse-sensitivity

**类型**：`float`  
**说明**：鼠标灵敏度  
**默认值**：`0.002`  
**示例**：
```json
"mouse-sensitivity": 0.002
```

### render-speed-ratio

**类型**：`integer`  
**说明**：粒子运动速度倍率  
**说明**：
- 值越大，粒子运动越慢
- `1` 表示原速
- 用于控制动画播放速度

**示例**：
```json
"render-speed-ratio": 4
```

### particle-shift

**类型**：`array of float`（3 个元素）  
**说明**：所有粒子的整体位移  
**示例**：
```json
"particle-shift": [0.0, 0.0, 0.0]
```

### particle-scale

**类型**：`array of float`（3 个元素）  
**说明**：所有粒子的整体缩放  
**示例**：
```json
"particle-scale": [1.0, 1.0, 1.0]
```

## 完整配置示例

```json
{
  "series-path": "../files/",
  "series-name": "particle_mesh-short.vtk.series",
  "cache-path": "../cache/",
  "cache": false,
  "debug-mode": false,
  "cache-process-thread-count": 8,
  "roughs": [
    {"albedo": [0.65, 0.05, 0.05]},
    {"albedo": [0.73, 0.73, 0.73]},
    {"albedo": [0.12, 0.45, 0.15]},
    {"albedo": [0.70, 0.60, 0.50]}
  ],
  "metals": [
    {"albedo": [0.8, 0.85, 0.88], "fuzz": 0.0}
  ],
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
  ],
  "triangles": [],
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
}
```

## 配置验证

程序启动时会验证配置文件：
- JSON 格式检查
- 必需字段检查
- 类型检查
- 平台兼容性检查（如 D3D11/D3D12 仅在 Windows）

如果配置有误，程序会输出错误信息并退出。

## 配置技巧

1. **路径使用相对路径**：便于项目迁移
2. **首次运行生成缓存**：设置 `"cache": true`，然后改回 `false`
3. **调整线程数**：根据 CPU 核心数设置 `cache-process-thread-count`
4. **测试不同 API**：尝试不同的图形 API 以找到最佳性能
5. **相机参数调优**：根据场景大小调整相机参数

## 常见配置错误

### 错误 1：路径不存在

确保 `series-path` 和 `cache-path` 目录存在。

### 错误 2：材质索引越界

确保 `mat-index` 在对应材质数组的有效范围内。

### 错误 3：Linux 使用 D3D

Linux 不支持 Direct3D，使用 `"OGL"` 或 `"VK"`。

### 错误 4：数组格式错误

确保数组格式正确，例如 `[0.0, 0.0, 0.0]` 而不是 `[0.0,0.0,0.0]`（虽然两者都有效，但建议使用空格）。

## 参考

- [使用指南](usage.md) - 使用示例
- [技术细节](technical-details.md) - 实现原理

