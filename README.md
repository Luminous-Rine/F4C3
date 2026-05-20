# F4C3 — 本地人脸考勤系统

基于 YOLO ONNX + InsightFace MobileFaceNet 的边缘端人脸考勤方案。
无 PyTorch 依赖，可直接部署到 Raspberry Pi 4。

---

## 目录结构

```
F4C3/
├── register.py          # 模块一：注册员工人脸特征
├── attendance.py        # 模块二：实时摄像头考勤
├── logger.py            # 模块三：打卡 CSV 记录器
├── face_detector.py     # 共享：YOLO ONNX 检测器
├── face_recognizer.py   # 共享：InsightFace 特征提取
├── requirements.txt
├── scripts/
│   └── prepare_models.py  # ONNX 模型下载/转换工具
├── models/              # 存放 yolov8n-face.onnx
├── insightface_model/   # InsightFace 自动缓存（首次运行下载）
├── dataset/             # 注册用照片（子目录名=员工姓名）
│   ├── 张三/
│   └── 李四/
├── logs/
│   └── attendance_log.csv
└── face_database.pkl    # 注册后生成
```

---

## 快速开始

### 第一步：安装依赖

```bash
# macOS 开发机
pip install -r requirements.txt
```

### 第二步：准备 YOLO face ONNX 模型

```bash
# 自动下载 yolov8n-face.pt 并转换为 ONNX（仅需在 macOS 执行一次）
python scripts/prepare_models.py
```

生成的 `models/yolov8n-face.onnx` 复制到树莓派同路径即可。

> **提示**：若无法访问 GitHub，可手动下载：
> https://github.com/akanametov/yolo-face/releases/tag/v0.0.0

### 第三步：准备数据集

```
dataset/
├── 张三/
│   ├── 01.jpg  （建议 10~20 张，不同光线/角度）
│   └── ...
└── 李四/
    └── ...
```

### 第四步：注册员工

```bash
python register.py
# 或指定参数：
python register.py --dataset ./dataset --db ./face_database.pkl
```

> 首次运行会自动下载 InsightFace `buffalo_s` 模型（约 85MB）。

### 第五步：启动考勤

```bash
# 正常模式（有显示器）
python attendance.py

# 无头模式（Raspberry Pi SSH 连接）
python attendance.py --no-display

# 完整参数示例
python attendance.py \
  --threshold 0.45 \
  --skip 2 \
  --cooldown 5 \
  --log ./logs/attendance_log.csv
```

---

## 参数说明

### register.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | `./dataset` | 数据集根目录 |
| `--db` | `./face_database.pkl` | 输出特征库 |
| `--model` | `./models/yolov8n-face.onnx` | YOLO 模型（不存在自动回退 InsightFace 检测）|
| `--min-face` | `40` | 最小有效人脸边长（px）|
| `--overwrite` | `False` | 覆盖已有记录 |

### attendance.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--threshold` | `0.45` | 余弦相似度识别阈值 |
| `--skip` | `2` | 跳帧数（RPi4 推荐 2~3）|
| `--cooldown` | `5` | 打卡防抖间隔（分钟）|
| `--no-display` | `False` | 无头模式 |
| `--camera` | `0` | 摄像头索引 |

---

## 树莓派 4 部署

```bash
# 1. 安装无头版 OpenCV（避免 GUI 依赖）
pip install opencv-python-headless onnxruntime insightface numpy

# 2. 复制文件到 RPi（不需要 ultralytics / PyTorch）
scp -r face_detector.py face_recognizer.py \
        register.py attendance.py logger.py \
        models/ face_database.pkl \
        insightface_model/ \
        pi@raspberrypi.local:~/f4c3/

# 3. 启动无头考勤
python attendance.py --no-display --skip 3
```

---

## CSV 日志格式

```csv
姓名,日期,打卡时间
张三,2026-05-20,09:02:15
李四,2026-05-20,09:05:43
```
