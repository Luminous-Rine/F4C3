"""
scripts/prepare_models.py
──────────────────────────────────────────────────────────────
功能：将 YOLOv8n-face 权重（.pt）转换为 ONNX 格式，
      以便在 Raspberry Pi 4 上通过 onnxruntime-arm64 推理。

【在 macOS 开发机上运行此脚本，生成的 .onnx 复制到树莓派】

用法：
    # 方案 A（推荐）：直接使用社区预训练的 yolov8n-face.pt
    python scripts/prepare_models.py

    # 方案 B：指定已有的 pt 文件
    python scripts/prepare_models.py --pt path/to/yolov8n-face.pt

    # 仅校验已有 onnx 是否可推理
    python scripts/prepare_models.py --check-only
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path


# ── 模型权重下载源（按优先级） ────────────────────────────────
MODEL_URLS = {
    "yolov8n-face.pt": (
        "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/"
        "yolov8n-face.pt"
    ),
}

MODELS_DIR = Path("./models")


def download_with_progress(url: str, dest: Path) -> None:
    """带进度显示的文件下载"""
    print(f"  Downloading: {url}")
    print(f"  Saving to  : {dest}")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            bar = "█" * int(pct // 4) + "░" * (25 - int(pct // 4))
            print(f"\r  [{bar}] {pct:.1f}%  ({downloaded//1024}KB)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook)
    print()


def export_to_onnx(pt_path: Path, onnx_path: Path) -> None:
    """使用 ultralytics 将 .pt 导出为 .onnx"""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[Error] 需要安装 ultralytics：pip install ultralytics")
        sys.exit(1)

    print(f"\n[Export] {pt_path} → {onnx_path}")
    model = YOLO(str(pt_path))
    # half=False 保证 RPi4 ARM CPU 兼容
    model.export(
        format="onnx",
        imgsz=640,
        simplify=True,
        dynamic=False,
        half=False,
        opset=12,
    )

    # ultralytics 默认导出到同目录同名 .onnx
    default_out = pt_path.with_suffix(".onnx")
    if default_out.exists() and default_out != onnx_path:
        default_out.rename(onnx_path)
    print(f"[Export] ✅ ONNX 模型已保存：{onnx_path}")


def check_onnx(onnx_path: Path) -> bool:
    """快速验证 ONNX 模型可被 onnxruntime 加载"""
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        inp = sess.get_inputs()[0]
        dummy = np.zeros([1, 3, 640, 640], dtype=np.float32)
        out = sess.run(None, {inp.name: dummy})
        print(f"[Check] ✅ 模型验证通过，输出 shape: {[o.shape for o in out]}")
        return True
    except Exception as exc:
        print(f"[Check] ❌ 模型验证失败: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="准备 YOLOv8n-face ONNX 模型")
    parser.add_argument("--pt", default=None,
                        help="指定已有的 .pt 权重文件路径（不指定则自动下载）")
    parser.add_argument("--check-only", action="store_true",
                        help="仅校验已有 ONNX 模型，不下载不转换")
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    onnx_path = MODELS_DIR / "yolov8n-face.onnx"

    # ── 仅校验模式 ────────────────────────────
    if args.check_only:
        if not onnx_path.exists():
            print(f"[Error] 找不到 {onnx_path}")
            sys.exit(1)
        check_onnx(onnx_path)
        return

    # ── 已存在 ONNX ───────────────────────────
    if onnx_path.exists():
        print(f"[Info] 检测到已有 ONNX 模型: {onnx_path}")
        check_onnx(onnx_path)
        return

    # ── 确定 pt 来源 ──────────────────────────
    if args.pt:
        pt_path = Path(args.pt)
        if not pt_path.exists():
            print(f"[Error] 指定的 pt 文件不存在: {pt_path}")
            sys.exit(1)
    else:
        pt_path = MODELS_DIR / "yolov8n-face.pt"
        if not pt_path.exists():
            print("\n[Download] 开始下载 yolov8n-face.pt ...")
            url = MODEL_URLS["yolov8n-face.pt"]
            download_with_progress(url, pt_path)
        else:
            print(f"[Info] 使用已有 pt 文件: {pt_path}")

    # ── 导出 ONNX ─────────────────────────────
    export_to_onnx(pt_path, onnx_path)
    check_onnx(onnx_path)

    print(f"\n🎉 准备完成！")
    print(f"   将 {onnx_path} 复制到树莓派的 models/ 目录即可。")
    print(f"   RPi4 只需安装 onnxruntime，无需 PyTorch。")


if __name__ == "__main__":
    main()
