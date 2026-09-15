"""
ArcFace ONNX embedding extraction. Decoupled from the detector;
operates only on an already-aligned 112x112 image.
"""
import numpy as np
import onnxruntime as ort
import cv2
from typing import Tuple


class ArcFaceEmbedderONNX:
    def __init__(
        self,
        model_path: str = "models/embedder_arcface.onnx",
        input_size: Tuple[int, int] = (112, 112),
    ):
        self.in_w, self.in_h = input_size
        self.sess = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

    def embed(self, aligned_bgr_112: np.ndarray) -> np.ndarray:
        """Input: aligned 112x112 BGR face. Output: L2-normalized 512-D vector."""
        if aligned_bgr_112.shape[:2] != (self.in_h, self.in_w):
            aligned_bgr_112 = cv2.resize(aligned_bgr_112, (self.in_w, self.in_h))
        rgb = cv2.cvtColor(aligned_bgr_112, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
        x = np.transpose(rgb, (2, 0, 1))[None, ...]
        y = self.sess.run([self.out_name], {self.in_name: x})[0]
        emb = y.reshape(-1).astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-12)