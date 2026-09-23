# src/expressions.py
"""
Facial expression analysis: blink, smile, frown.

Uses OpenCV's own Facemark API (cv2.face.FacemarkLBF, 68-point dlib-style
landmarks) instead of MediaPipe -- this only requires opencv-contrib-python
(no separate framework, no MediaPipe wheel/version headaches). It runs on
the already-detected face box from YuNet (src/detect_landmarks.py), so
identity/recognition is untouched -- this module only adds expression
signals on top.

Setup (one-time):
    1. pip uninstall opencv-python opencv-python-headless -y
       pip install opencv-contrib-python
       (only ONE of opencv-python / opencv-contrib-python may be installed
       at a time -- they both provide the "cv2" module and will conflict.
       opencv-contrib-python is a strict superset, so nothing is lost.)
    2. Download the pretrained LBF model (~54 MB) to models/lbfmodel.yaml:
       PowerShell:
         curl.exe -L -o models\\lbfmodel.yaml `
           https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml

Landmark indices below follow the standard 68-point iBUG layout (the same
one dlib uses), which is what this pretrained model was trained on:
    0-16   jaw line
    17-21  left eyebrow      22-26  right eyebrow
    27-35  nose
    36-41  left eye          42-47  right eye
    48-67  mouth (48-59 outer, 60-67 inner)

Heuristics:
  - Blink: Eye Aspect Ratio (EAR), Soukupova & Cech. Drops sharply when an
    eye closes. A blink is counted when EAR stays below threshold for a
    short run of consecutive frames and then recovers.
  - Smile / Frown: mouth width relative to face width (wide mouth = smile
    candidate) combined with mouth-corner elevation relative to the mouth
    center (corners up = smile, corners down = frown). Heuristic, not a
    trained classifier -- tune the thresholds below if needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


# ============================================================
# LANDMARK INDICES (68-point iBUG layout)
# ============================================================

LEFT_EYE_IDX = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_IDX = [42, 43, 44, 45, 46, 47]

MOUTH_LEFT_IDX = 48
MOUTH_RIGHT_IDX = 54
MOUTH_TOP_IDX = 51
MOUTH_BOTTOM_IDX = 57

JAW_LEFT_IDX = 0
JAW_RIGHT_IDX = 16


# ============================================================
# TUNABLE THRESHOLDS
# ============================================================

EAR_BLINK_THRESHOLD = 0.21
EAR_CONSEC_FRAMES = 2

SMILE_WIDTH_RATIO = 0.42
SMILE_CORNER_RISE_PX = 1.5
FROWN_CORNER_DROP_PX = 1.5

DEFAULT_MODEL_PATH = "models/lbfmodel.yaml"


def _dist(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


def _eye_aspect_ratio(pts: np.ndarray) -> float:
    """pts: (6,2) in iBUG order [outer, upper1, upper2, inner, lower2, lower1]
    matching indices 36-41 / 42-47 directly."""
    p1, p2, p3, p4, p5, p6 = pts
    vertical = _dist(p2, p6) + _dist(p3, p5)
    horizontal = _dist(p1, p4)
    if horizontal <= 1e-6:
        return 0.0
    return vertical / (2.0 * horizontal)


@dataclass
class ExpressionResult:
    ear_left: float = 0.0
    ear_right: float = 0.0
    ear_avg: float = 0.0
    eyes_closed: bool = False
    blink_event: bool = False
    smile: bool = False
    frown: bool = False
    landmarks_found: bool = False


@dataclass
class _BlinkState:
    closed_run: int = 0
    total_blinks: int = 0


class ExpressionAnalyzer:
    """
    Stateful per-target analyzer. Create one instance and call reset()
    whenever the tracked identity changes, so blink counting stays
    meaningful per-person.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"LBF model not found at '{model_path}'. Download it with:\n"
                f'  curl.exe -L -o {model_path} '
                f"https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml"
            )
        self._facemark = cv2.face.createFacemarkLBF()
        self._facemark.loadModel(model_path)
        self._blink = _BlinkState()

    def reset(self):
        """Call when the tracked target changes so blink count restarts."""
        self._blink = _BlinkState()

    @property
    def blink_count(self) -> int:
        return self._blink.total_blinks

    def close(self):
        pass  # nothing to release explicitly for FacemarkLBF

    def analyze(self, frame_bgr: np.ndarray,
                box: Tuple[int, int, int, int]) -> ExpressionResult:
        """
        box: (x1, y1, x2, y2) face box in frame_bgr coordinates, e.g. from
        YuNet. FacemarkLBF wants (x, y, w, h) rects, so it's converted here.
        """
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return ExpressionResult()

        faces_rect = np.array([[x1, y1, w, h]], dtype=np.int32)

        ok, landmarks = self._facemark.fit(frame_bgr, faces_rect)
        if not ok or landmarks is None or len(landmarks) == 0:
            return ExpressionResult(landmarks_found=False)

        pts = landmarks[0].reshape(-1, 2).astype(np.float32)  # (68, 2)

        # ---- Blink (EAR) ----
        left_ear = _eye_aspect_ratio(pts[LEFT_EYE_IDX])
        right_ear = _eye_aspect_ratio(pts[RIGHT_EYE_IDX])
        avg_ear = (left_ear + right_ear) / 2.0

        eyes_closed = bool(avg_ear < EAR_BLINK_THRESHOLD)
        blink_event = False

        if eyes_closed:
            self._blink.closed_run += 1
        else:
            if self._blink.closed_run >= EAR_CONSEC_FRAMES:
                self._blink.total_blinks += 1
                blink_event = True
            self._blink.closed_run = 0

        # ---- Smile / Frown ----
        mouth_left = pts[MOUTH_LEFT_IDX]
        mouth_right = pts[MOUTH_RIGHT_IDX]
        mouth_top = pts[MOUTH_TOP_IDX]
        mouth_bottom = pts[MOUTH_BOTTOM_IDX]
        jaw_left = pts[JAW_LEFT_IDX]
        jaw_right = pts[JAW_RIGHT_IDX]

        face_width = _dist(jaw_left, jaw_right)
        mouth_width = _dist(mouth_left, mouth_right)
        mouth_center_y = (mouth_top[1] + mouth_bottom[1]) / 2.0
        corner_avg_y = (mouth_left[1] + mouth_right[1]) / 2.0

        width_ratio = mouth_width / face_width if face_width > 1e-6 else 0.0
        corner_rise = mouth_center_y - corner_avg_y  # image y grows downward

        smile = bool(width_ratio > SMILE_WIDTH_RATIO and corner_rise > SMILE_CORNER_RISE_PX)
        frown = bool(corner_rise < -FROWN_CORNER_DROP_PX and not smile)

        return ExpressionResult(
            ear_left=left_ear,
            ear_right=right_ear,
            ear_avg=avg_ear,
            eyes_closed=eyes_closed,
            blink_event=blink_event,
            smile=smile,
            frown=frown,
            landmarks_found=True,
        )