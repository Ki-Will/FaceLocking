"""
YuNet unified face detection + 5-point landmark extraction.
Replaces the Haar + MediaPipe dual-model approach in the original haar_5pt.py.
"""
from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class FaceDet:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray  # (5,2) float32: left eye, right eye, nose tip, left mouth, right mouth


class YuNetDetector:
    def __init__(
        self,
        model_path: str = "models/face_detection_yunet_2023mar.onnx",
        input_size: Tuple[int, int] = (320, 320),
        score_thresh: float = 0.9,
        nms_thresh: float = 0.3,
        top_k: int = 5000,
    ):
        self.detector = cv2.FaceDetectorYN.create(
            model_path, "", input_size,
            score_thresh, nms_thresh, top_k,
        )
        self.input_size = input_size

    def detect(self, frame_bgr: np.ndarray, max_faces: int = 5) -> List[FaceDet]:
        h, w = frame_bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _, results = self.detector.detect(frame_bgr)

        if results is None:
            return []

        out: List[FaceDet] = []
        for r in results[:max_faces]:
            # r format: [x, y, w, h, 5x(x,y), score]
            x, y, bw, bh = r[:4].astype(int)
            kps = r[4:14].reshape(5, 2).astype(np.float32)
            score = float(r[14])

            # Enforce left/right ordering
            if kps[0, 0] > kps[1, 0]:
                kps[[0, 1]] = kps[[1, 0]]
            if kps[3, 0] > kps[4, 0]:
                kps[[3, 4]] = kps[[4, 3]]

            # Clip to image bounds
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w - 1, x + bw), min(h - 1, y + bh)

            out.append(FaceDet(x1, y1, x2, y2, score, kps))
        return out