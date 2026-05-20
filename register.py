"""
register.py  ── 模块一：本地特征库注册工具

用法：
    python register.py [--dataset ./dataset] [--db face_database.pkl]
                       [--model models/yolov8n-face.onnx] [--min-face 40]

流程：
  1. 遍历 dataset/  各子文件夹（子文件夹名 = 员工姓名）
  2. YOLO 检测每张图片中的人脸 → 取置信度最高的一张
  3. InsightFace norm_crop 对齐 → MobileFaceNet 提取 512-d embedding
  4. 对每人的所有 embedding 取均值并 L2 归一化 → 存入 face_database.pkl
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

from face_detector import YOLOFaceDetector
from face_recognizer import FaceRecognizer


# ─────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ─────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────

def load_database(db_path: str) -> dict:
    """加载已有特征库（不存在则返回空 dict）"""
    if os.path.exists(db_path):
        with open(db_path, "rb") as f:
            db = pickle.load(f)
        print(f"[Register] 已加载现有特征库，共 {len(db)} 人：{list(db.keys())}")
        return db
    return {}


def save_database(db: dict, db_path: str) -> None:
    """序列化特征库到 pkl"""
    with open(db_path, "wb") as f:
        pickle.dump(db, f)
    print(f"[Register] 特征库已保存到 {db_path}，共 {len(db)} 人。")


def process_person(
    name: str,
    img_paths: list[Path],
    detector: YOLOFaceDetector,
    recognizer: FaceRecognizer,
    min_face_px: int,
    use_fallback: bool,
) -> np.ndarray | None:
    """
    处理单个员工的所有图片，返回该人的平均特征向量（L2归一化）。

    Parameters
    ----------
    use_fallback : 若为 True，YOLO 模型不可用时启用 InsightFace 内置检测
    """
    embeddings = []

    for img_path in img_paths:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"  [Skip] 无法读取图像: {img_path.name}")
            continue

        if use_fallback:
            # InsightFace 完整流水线（register 时用，accuracy 优先）
            faces = recognizer.embed_with_insightface(bgr)
            if not faces:
                print(f"  [Skip] 未检测到人脸: {img_path.name}")
                continue
            # 选面积最大的人脸
            best = max(
                faces,
                key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]),
            )
            emb = best["embedding"]
        else:
            dets = detector.detect(bgr)
            if not dets:
                print(f"  [Skip] 未检测到人脸: {img_path.name}")
                continue

            # 过滤过小人脸
            dets = [
                d for d in dets
                if (d["bbox"][2] - d["bbox"][0]) >= min_face_px
                and (d["bbox"][3] - d["bbox"][1]) >= min_face_px
            ]
            if not dets:
                print(f"  [Skip] 人脸过小（<{min_face_px}px）: {img_path.name}")
                continue

            best = max(dets, key=lambda d: d["conf"])
            emb = recognizer.align_and_embed(bgr, best)
            if emb is None:
                print(f"  [Skip] 特征提取失败: {img_path.name}")
                continue

        embeddings.append(emb)
        print(f"  [OK]   {img_path.name}")

    if not embeddings:
        return None

    mean_emb = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(mean_emb)
    return mean_emb / norm if norm > 1e-6 else mean_emb


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="人脸特征库注册工具")
    parser.add_argument("--dataset", default="./dataset",
                        help="数据集根目录，子文件夹名为员工姓名")
    parser.add_argument("--db", default="./face_database.pkl",
                        help="输出特征库文件路径")
    parser.add_argument("--model", default="./models/yolov8n-face.onnx",
                        help="YOLO face ONNX 模型路径")
    parser.add_argument("--insightface-root", default="./insightface_model",
                        help="InsightFace 模型缓存目录")
    parser.add_argument("--min-face", type=int, default=40,
                        help="最小有效人脸边长（像素），默认 40")
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖已有记录（默认追加/更新）")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        print(f"[Error] 数据集目录不存在: {dataset_dir}")
        sys.exit(1)

    # ── 初始化模型 ───────────────────────────
    use_fallback = not os.path.exists(args.model)
    if use_fallback:
        print(f"[Warn] 未找到 YOLO 模型 ({args.model})，"
              "回退使用 InsightFace 内置检测器（精度较高但速度稍慢）。")
        detector = None
    else:
        print(f"[Info] 加载 YOLO 模型: {args.model}")
        detector = YOLOFaceDetector(
            model_path=args.model,
            conf_thresh=0.45,
            iou_thresh=0.45,
        )

    print("[Info] 初始化 InsightFace recognition 模型（首次运行会自动下载 ~85MB）...")
    recognizer = FaceRecognizer(model_root=args.insightface_root)

    # ── 加载/新建特征库 ──────────────────────
    db = {} if args.overwrite else load_database(args.db)

    # ── 遍历员工子目录 ───────────────────────
    person_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    if not person_dirs:
        print(f"[Error] 数据集目录下没有子文件夹: {dataset_dir}")
        sys.exit(1)

    print(f"\n[Register] 发现 {len(person_dirs)} 位员工，开始注册...\n")

    for person_dir in person_dirs:
        name = person_dir.name
        img_paths = sorted([
            p for p in person_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_EXT
        ])

        if not img_paths:
            print(f"[{name}] 目录为空，跳过。")
            continue

        print(f"[{name}] 处理 {len(img_paths)} 张图片...")
        mean_emb = process_person(
            name=name,
            img_paths=img_paths,
            detector=detector,
            recognizer=recognizer,
            min_face_px=args.min_face,
            use_fallback=use_fallback,
        )

        if mean_emb is None:
            print(f"[{name}] ⚠️  未能提取任何有效 embedding，跳过注册。\n")
            continue

        db[name] = mean_emb
        print(f"[{name}] ✅ 注册成功，embedding shape: {mean_emb.shape}\n")

    # ── 保存 ──────────────────────────────────
    if not db:
        print("[Error] 特征库为空，请检查数据集。")
        sys.exit(1)

    save_database(db, args.db)
    print(f"\n✅ 注册完成！已注册员工：{list(db.keys())}")


if __name__ == "__main__":
    main()
