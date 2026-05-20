"""
attendance.py  ── 模块二：实时考勤推理流水线

用法：
    python attendance.py [--db face_database.pkl] [--model models/yolov8n-face.onnx]
                         [--camera 0] [--threshold 0.45] [--skip 2]

流程：
  1. 捕获：640×480 摄像头帧
  2. 检测：YOLO ONNX → 人脸 bbox + keypoints
  3. 裁剪/对齐：InsightFace norm_crop → 112×112
  4. 特征：MobileFaceNet → 512-d embedding
  5. 比对：余弦相似度 vs face_database.pkl
  6. 判定：相似度 > threshold → 打卡；否则 "Unknown"
  7. 记录：logger.py 防抖写 CSV
"""

import argparse
import os
import pickle
import sys
import time

import cv2
import numpy as np

from face_detector import YOLOFaceDetector
from face_recognizer import FaceRecognizer
from logger import AttendanceLogger


# ─────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────

FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480
WIN_TITLE     = "F4C3 Attendance System  |  Press 'q' to quit"

COLOR_KNOWN   = (0,  220,  80)   # 绿
COLOR_UNKNOWN = (0,   80, 220)   # 红橙
COLOR_TEXT_BG = (20,  20,  20)


# ─────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────

def load_database(db_path: str) -> dict:
    """加载特征库，返回 {name: embedding(512,)} dict"""
    if not os.path.exists(db_path):
        print(f"[Error] 特征库不存在: {db_path}")
        print("       请先运行 register.py 完成注册。")
        sys.exit(1)
    with open(db_path, "rb") as f:
        db = pickle.load(f)
    print(f"[Attendance] 已加载特征库，共 {len(db)} 人：{list(db.keys())}")
    return db


def identify(
    embedding: np.ndarray,
    database: dict,
    threshold: float,
) -> tuple[str, float]:
    """
    余弦相似度匹配。

    Returns
    -------
    (name, similarity)  name 为 "Unknown" 时 similarity 为最高分
    """
    best_name = "Unknown"
    best_sim  = -1.0

    for name, ref_emb in database.items():
        sim = float(np.dot(embedding, ref_emb))   # 均已 L2 归一化，点积即余弦相似度
        if sim > best_sim:
            best_sim  = sim
            best_name = name

    if best_sim < threshold:
        best_name = "Unknown"

    return best_name, best_sim


def draw_result(
    frame: np.ndarray,
    bbox: list[int],
    name: str,
    sim: float,
    logged: bool,
) -> None:
    """在帧上绘制检测框和标签"""
    x1, y1, x2, y2 = bbox
    color = COLOR_KNOWN if name != "Unknown" else COLOR_UNKNOWN

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if name != "Unknown":
        label = f"{name}  {sim:.2f}"
        badge = "✓ 已打卡" if logged else ""
    else:
        label = f"Unknown  {sim:.2f}"
        badge = ""

    # 标签背景
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), COLOR_TEXT_BG, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    if badge:
        cv2.putText(frame, badge, (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_KNOWN, 1, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, fps: float, face_count: int) -> None:
    """左上角显示帧率和人脸数量"""
    info = f"FPS: {fps:.1f}  Faces: {face_count}"
    cv2.putText(frame, info, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────
# 主循环
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="实时人脸考勤系统")
    parser.add_argument("--db", default="./face_database.pkl",
                        help="特征库路径")
    parser.add_argument("--model", default="./models/yolov8n-face.onnx",
                        help="YOLO face ONNX 模型路径")
    parser.add_argument("--insightface-root", default="./insightface_model",
                        help="InsightFace 模型缓存目录")
    parser.add_argument("--camera", type=int, default=0,
                        help="摄像头索引，默认 0")
    parser.add_argument("--threshold", type=float, default=0.45,
                        help="识别相似度阈值，默认 0.45")
    parser.add_argument("--skip", type=int, default=2,
                        help="跳帧数：每 N 帧处理 1 帧，默认 2（RPi4 推荐 2~3）")
    parser.add_argument("--log", default="./logs/attendance_log.csv",
                        help="CSV 打卡日志路径")
    parser.add_argument("--cooldown", type=int, default=5,
                        help="同一人打卡冷却时间（分钟），默认 5")
    parser.add_argument("--no-display", action="store_true",
                        help="无头模式，不显示画面（Headless RPi4 使用）")
    args = parser.parse_args()

    # ── 初始化模型 ───────────────────────────
    use_fallback = not os.path.exists(args.model)
    if use_fallback:
        print(f"[Warn] 未找到 YOLO 模型 ({args.model})，"
              "回退使用 InsightFace 内置检测器。")
        detector = None
    else:
        print(f"[Info] 加载 YOLO 检测模型: {args.model}")
        detector = YOLOFaceDetector(
            model_path=args.model,
            conf_thresh=0.45,
            iou_thresh=0.45,
            num_threads=4,
        )

    print("[Info] 加载 InsightFace 识别模型...")
    recognizer = FaceRecognizer(model_root=args.insightface_root)

    database = load_database(args.db)
    logger   = AttendanceLogger(log_file=args.log, cooldown_minutes=args.cooldown)

    # ── 摄像头初始化 ─────────────────────────
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"[Error] 无法打开摄像头 (index={args.camera})")
        sys.exit(1)

    print(f"\n[Attendance] 系统启动 ▶  阈值={args.threshold}  跳帧={args.skip}")
    if not args.no_display:
        print(f"             按 'q' 退出。")

    # ── 主循环变量 ───────────────────────────
    frame_idx    = 0
    last_dets    = []          # 上一次检测结果（跳帧时复用）
    last_results = []          # [(name, sim, logged), ...] 与 last_dets 对应
    fps_t        = time.time()
    fps_counter  = 0
    fps_display  = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[Error] 无法读取摄像头帧，退出。")
                break

            frame_idx   += 1
            fps_counter += 1

            # ── FPS 统计（每 30 帧更新一次） ───
            if fps_counter >= 30:
                elapsed = time.time() - fps_t
                fps_display = fps_counter / elapsed if elapsed > 0 else 0
                fps_t       = time.time()
                fps_counter = 0

            # ── 跳帧逻辑 ────────────────────────
            do_infer = (frame_idx % (args.skip + 1) == 0)

            if do_infer:
                # 步骤 2：人脸检测
                if use_fallback:
                    raw_faces  = recognizer.embed_with_insightface(frame)
                    detections = raw_faces          # 含 embedding 字段
                else:
                    detections = detector.detect(frame)

                last_dets    = detections
                last_results = []

                # 步骤 3/4/5/6/7：每个人脸依次处理
                for det in detections:
                    if use_fallback:
                        emb = det["embedding"]
                    else:
                        emb = recognizer.align_and_embed(frame, det)

                    if emb is None:
                        last_results.append(("Unknown", 0.0, False))
                        continue

                    name, sim  = identify(emb, database, args.threshold)
                    logged     = logger.log(name)
                    last_results.append((name, sim, logged))

            # ── 可视化 ──────────────────────────
            if not args.no_display:
                for det, res in zip(last_dets, last_results):
                    name, sim, logged = res
                    bbox = det["bbox"]
                    draw_result(frame, bbox, name, sim, logged)

                draw_hud(frame, fps_display, len(last_dets))
                cv2.imshow(WIN_TITLE, frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # 无头模式：打印识别结果
                for det, res in zip(last_dets, last_results):
                    name, sim, logged = res
                    if do_infer and name != "Unknown":
                        print(f"[Detect] {name}  sim={sim:.3f}  logged={logged}")

    except KeyboardInterrupt:
        print("\n[Attendance] 用户中断，退出。")

    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()

        summary = logger.today_summary()
        print(f"\n[Summary] 今日共 {len(summary)} 条打卡记录：")
        for row in summary:
            print(f"  {row.get('姓名')}  {row.get('打卡时间')}")


if __name__ == "__main__":
    main()
