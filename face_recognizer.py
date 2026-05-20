"""
face_recognizer.py
InsightFace MobileFaceNet 特征提取封装。
内部全链路使用 onnxruntime（无 PyTorch 依赖）。

流程：YOLO-bbox / keypoints → norm_crop (112×112) → embedding (512-d, L2归一化)
"""

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align


class FaceRecognizer:
    """
    基于 InsightFace buffalo_s 的人脸特征提取器。

    Parameters
    ----------
    model_root   : InsightFace 模型缓存根目录，默认 './insightface_model'
    model_name   : InsightFace 模型包，默认 'buffalo_s'（轻量，RPi4 友好）
    det_size     : InsightFace 内置检测器输入尺寸（仅在 fallback 时使用）
    """

    def __init__(
        self,
        model_root: str = "./insightface_model",
        model_name: str = "buffalo_s",
        det_size: tuple = (320, 320),
    ):
        self._app = FaceAnalysis(
            name=model_name,
            root=model_root,
            allowed_modules=["detection", "recognition"],
        )
        # ctx_id=-1 → CPU（RPi4 无 GPU）
        self._app.prepare(ctx_id=-1, det_size=det_size)

        # 找到 recognition 模型实例（带 get_feat 方法）
        self._rec_model = None
        for model in self._app.models.values():
            if hasattr(model, "get_feat"):
                self._rec_model = model
                break

        if self._rec_model is None:
            raise RuntimeError(
                "未找到 InsightFace recognition 模型，"
                "请确认 buffalo_s 已正确下载到 model_root。"
            )

    # ── 公开接口 ────────────────────────────────

    def get_embedding_from_aligned(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        从已对齐的 112×112 BGR 图像提取 512-d L2 归一化特征向量。

        Parameters
        ----------
        aligned_face : np.ndarray, shape (112, 112, 3), BGR

        Returns
        -------
        np.ndarray, shape (512,)
        """
        face_112 = cv2.resize(aligned_face, (112, 112))
        feat = self._rec_model.get_feat([face_112])
        embedding = feat[0] if feat.ndim == 2 else feat
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 1e-6 else embedding

    def align_and_embed(
        self,
        bgr_img: np.ndarray,
        det: dict,
    ) -> np.ndarray | None:
        """
        给定原图和一条 YOLO 检测结果，完成对齐 + 特征提取。

        优先使用 YOLO 关键点做 ArcFace 仿射对齐；
        无关键点时回退到 bbox 直接裁剪缩放。

        Parameters
        ----------
        bgr_img : 原始 BGR 帧
        det     : YOLOFaceDetector.detect() 返回的单条结果 dict

        Returns
        -------
        np.ndarray shape (512,) 或 None（对齐/裁剪失败时）
        """
        try:
            kps = det.get("kps")
            if kps is not None:
                aligned = face_align.norm_crop(bgr_img, landmark=kps)
            else:
                x1, y1, x2, y2 = det["bbox"]
                crop = bgr_img[y1:y2, x1:x2]
                if crop.size == 0:
                    return None
                aligned = cv2.resize(crop, (112, 112))

            return self.get_embedding_from_aligned(aligned)
        except Exception as exc:
            print(f"[FaceRecognizer] align_and_embed 异常: {exc}")
            return None

    def embed_with_insightface(self, bgr_img: np.ndarray) -> list[dict]:
        """
        直接调用 InsightFace 内置检测+识别流水线（仅注册时用作 fallback）。

        Returns
        -------
        list of dict: [{'bbox', 'kps', 'embedding'}, ...]
        """
        faces = self._app.get(bgr_img)
        results = []
        for f in faces:
            emb = f.normed_embedding
            results.append({
                "bbox": f.bbox.astype(int).tolist(),
                "kps":  f.kps,
                "embedding": emb,
            })
        return results
