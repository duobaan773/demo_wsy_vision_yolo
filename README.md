# RoboMaster 2027 视觉考核阶段二（大一）

## 项目简介

本项目为 RoboMaster 2027 视觉组阶段二（大一）考核项目。

基于阶段一训练得到的 YOLO ONNX 模型，使用 **OpenCV DNN** 完成 C++ 部署，实现装甲板目标检测，并支持图片、视频以及 USB 摄像头三种输入方式。

目前已完成：

- YOLO ONNX 模型加载
- 图片检测
- 视频检测
- USB 摄像头检测
- 多目标检测
- 检测框绘制
- 类别显示
- 置信度显示
- 装甲板中心点计算
- FPS 与推理耗时统计
- 图片、视频检测结果保存

---

# 工程结构

```text
demo_wsy_vision_yolo
│
├── models
│   └── best.onnx
│
└── detect
    ├── assets/
    │
    ├── include/
    │   ├── detector.h
    │   ├── input_handler.h
    │   ├── timer.h
    │   └── visualizer.h
    │
    ├── results/
    │
    ├── src/
    │   ├── detector.cpp
    │   ├── input_handler.cpp
    │   ├── timer.cpp
    │   ├── visualizer.cpp
    │   └── main.cpp
    │
    ├── CMakeLists.txt
    └── README.md
```

---

# 开发环境

| 软件 | 版本 |
|------|------|
| Windows | 11 |
| C++ | C++17 |
| OpenCV | 4.13 |
| CMake | 3.20+ |
| MinGW | UCRT64 |
| 推理框架 | OpenCV DNN |

---

# 模型信息

模型名称：

```text
best.onnx
```

输入尺寸：

```text
1024 × 1024
```

检测类别：

```text
bluesb
redsb
blue3
red3
blue1
red1
```

---

# 编译

进入 detect 目录：

```bash
cd detect
```

创建 build：

```bash
mkdir build
```

生成工程：

```bash
cmake ..
```

编译：

```bash
cmake --build .
```

---

# 运行

## 图片

```bash
.\build\rm_detect.exe image assets/armor_test.jpg
```

---

## 视频

```bash
.\build\rm_detect.exe video assets/test.mp4
```

---

## USB 摄像头

```bash
.\build\rm_detect.exe camera 0
```

---

# 检测流程

```text
图片 / 视频 / 摄像头

        │

        ▼

LetterBox 预处理

        │

        ▼

OpenCV DNN 加载 ONNX

        │

        ▼

YOLO Forward 推理

        │

        ▼

输出解析

        │

        ▼

NMS 去除重复检测

        │

        ▼

计算装甲板中心点

        │

        ▼

绘制检测框
类别
置信度
中心点
FPS

        │

        ▼

显示检测结果
保存检测结果
```

---

# 已实现功能

| 功能 | 状态 |
|------|------|
| ONNX 模型部署 | ✅ |
| 图片检测 | ✅ |
| 视频检测 | ✅ |
| 摄像头检测 | ✅ |
| LetterBox 预处理 | ✅ |
| OpenCV DNN 推理 | ✅ |
| YOLO 后处理 | ✅ |
| NMS | ✅ |
| 多目标检测 | ✅ |
| 检测框绘制 | ✅ |
| 类别显示 | ✅ |
| 置信度显示 | ✅ |
| 中心点计算 | ✅ |
| FPS 统计 | ✅ |
| 推理耗时统计 | ✅ |
| 图片结果保存 | ✅ |
| 视频结果保存 | ✅ |

---

# 检测效果

建议在此处放置：

- 图片检测结果截图
- 视频检测结果截图

---

# Git 分支

```text
main
    阶段一（模型训练）

detect
    阶段二（C++部署）
```

---

# 作者

王姝怡25软件工程04
