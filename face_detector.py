"""
face_detector.py
YOLO-face ONNX 推理封装，兼容 YOLOv8-face 输出格式。
输出两类候选张量：
  - (N, 5)  : [cx, cy, w, h, conf]
  - (N, 20) : [cx, cy, w, h, conf, kp0x, kp0y, kp0v, ..., kp4x, kp4y, kp4v]

依赖：onnxruntime, numpy, opencv-python-headless
"""

import cv2
import numpy as np
import onnxruntime as ort


# ──────────────────────────────────────────────
# 内部工具函数
# ──────────────────────────────────────────────

def _letterbox(img: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
    """等比缩放 + 灰边填充，返回 (padded_img, ratio, (dw, dh))"""
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """CPU 版 NMS，返回保留框的索引数组"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return np.array(keep, dtype=np.int32)


# ──────────────────────────────────────────────
# 主类
# ──────────────────────────────────────────────

class YOLOFaceDetector:
    """
    基于 ONNX Runtime 的 YOLOv8-face 人脸检测器。

    Parameters
    ----------
    model_path   : ONNX 模型文件路径
    input_size   : 模型输入尺寸，默认 (640, 640)
    conf_thresh  : 置信度阈值，默认 0.45
    iou_thresh   : NMS IoU 阈值，默认 0.45
    num_threads  : onnxruntime 线程数（RPi4 建议设为 4）
    """

    def __init__(
        self,
        model_path: str,
        input_size: tuple = (640, 640),
        conf_thresh: float = 0.45,
        iou_thresh: float = 0.45,
        num_threads: int = 4,
    ):
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = num_threads
        sess_opts.inter_op_num_threads = num_threads
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            model_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

    # ── 公开接口 ────────────────────────────────

    def detect(self, bgr_img: np.ndarray):
        """
        检测图像中的所有人脸。

        Parameters
        ----------
        bgr_img : BGR 格式原始帧（任意分辨率）

        Returns
        -------
        list of dict，每个 dict 包含：
            'bbox'  : [x1, y1, x2, y2]  原图坐标（int）
            'conf'  : float  置信度
            'kps'   : np.ndarray shape (5,2) 或 None（无关键点时）
        """
        orig_h, orig_w = bgr_img.shape[:2]
        padded, ratio, (dw, dh) = _letterbox(bgr_img, self.input_size)

        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)

        raw = self._session.run(None, {self._input_name: blob})[0]  # (1, C, N) or (1, N, C)
        preds = np.squeeze(raw)  # (C, N) or (N, C)

        if preds.ndim == 1:
            return []

        # 统一为 (N, C) 格式
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T  # (C,N) → (N,C)

        has_kps = preds.shape[1] >= 20

        # 解析 bbox（中心格式）
        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        conf = preds[:, 4]

        mask = conf >= self.conf_thresh
        if mask.sum() == 0:
            return []

        cx, cy, bw, bh, conf = cx[mask], cy[mask], bw[mask], bh[mask], conf[mask]
        kps_raw = preds[mask, 5:20] if has_kps else None

        # 中心格式 → 角点格式（letterbox 空间）
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # 还原到原图坐标
        x1 = np.clip((x1 - dw) / ratio, 0, orig_w - 1)
        y1 = np.clip((y1 - dh) / ratio, 0, orig_h - 1)
        x2 = np.clip((x2 - dw) / ratio, 0, orig_w - 1)
        y2 = np.clip((y2 - dh) / ratio, 0, orig_h - 1)

        boxes = np.stack([x1, y1, x2, y2], axis=1)
        keep = _nms(boxes, conf, self.iou_thresh)

        results = []
        for i in keep:
            kps = None
            if kps_raw is not None:
                pts = kps_raw[i].reshape(5, 3)          # (5, [x, y, v])
                pts[:, 0] = np.clip((pts[:, 0] - dw) / ratio, 0, orig_w - 1)
                pts[:, 1] = np.clip((pts[:, 1] - dh) / ratio, 0, orig_h - 1)
                kps = pts[:, :2]                         # (5, 2)

            results.append({
                "bbox": boxes[i].astype(int).tolist(),
                "conf": float(conf[i]),
                "kps":  kps,
            })

        return results

    def crop_face(self, bgr_img: np.ndarray, det: dict,
                  pad_ratio: float = 0.1) -> np.ndarray:
        """
        根据检测结果裁剪人脸区域（含边距扩展）。
        当无关键点时使用此方法作为对齐备选。
        """
        x1, y1, x2, y2 = det["bbox"]
        h, w = bgr_img.shape[:2]
        pad_x = int((x2 - x1) * pad_ratio)
        pad_y = int((y2 - y1) * pad_ratio)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w - 1, x2 + pad_x)
        y2 = min(h - 1, y2 + pad_y)
        return bgr_img[y1:y2, x1:x2]
