"""
5-point affine alignment to ArcFace's 112x112 canonical template.
"""
from __future__ import annotations
import cv2
import numpy as np
from typing import Tuple


def _estimate_norm_5pt(kps_5x2: np.ndarray,
                       out_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
    """
    Build a 2x3 affine matrix mapping 5 keypoints to the ArcFace template.
    Keypoint order must be: [Leye, Reye, Nose, Lmouth, Rmouth].
    """
    k = kps_5x2.astype(np.float32)
    dst = np.array([
        [38.2946, 51.6963],   # left eye
        [73.5318, 51.5014],   # right eye
        [56.0252, 71.7366],   # nose
        [41.5493, 92.3655],   # left mouth
        [70.7299, 92.2041],   # right mouth
    ], dtype=np.float32)

    out_w, out_h = int(out_size[0]), int(out_size[1])
    if (out_w, out_h) != (112, 112):
        sx, sy = out_w / 112.0, out_h / 112.0
        dst = dst * np.array([sx, sy], dtype=np.float32)

    M, _ = cv2.estimateAffinePartial2D(k, dst, method=cv2.LMEDS)
    if M is None:
        M = cv2.getAffineTransform(
            np.array([k[0], k[1], k[2]], dtype=np.float32),
            np.array([dst[0], dst[1], dst[2]], dtype=np.float32),
        )
    return M.astype(np.float32)


def align_face_5pt(frame_bgr: np.ndarray,
                   kps_5x2: np.ndarray,
                   out_size: Tuple[int, int] = (112, 112)
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (aligned_bgr, affine_matrix M)."""
    M = _estimate_norm_5pt(kps_5x2, out_size=out_size)
    out_w, out_h = int(out_size[0]), int(out_size[1])
    aligned = cv2.warpAffine(
        frame_bgr, M, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return aligned, M