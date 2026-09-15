# src/recognize.py
"""
Recognition module + standalone demo.
Exposes `Recognizer` (embed + match) for reuse by track_with_recognition.py.

Run standalone:  python -m src.recognize
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .config import cfg
from .detect_landmarks import YuNetDetector, FaceDet
from .align import align_face_5pt
from .embed import ArcFaceEmbedderONNX


DB_PATH = Path("data/db/face_db.npz")


def load_db() -> Dict[str, np.ndarray]:
    if not DB_PATH.exists() or DB_PATH.stat().st_size == 0:
        return {}
    try:
        data = np.load(str(DB_PATH), allow_pickle=True)
        return {k: np.asarray(data[k], dtype=np.float32).reshape(-1) for k in data.files}
    except Exception as e:
        print(f"[load_db] {DB_PATH} unreadable ({e})")
        return {}


class Recognizer:
    """Encapsulates embedder + enrolled DB + matching."""

    def __init__(self):
        self.detector = YuNetDetector()
        self.embedder = ArcFaceEmbedderONNX(model_path=cfg.arc_face_model)
        self.threshold = cfg.recognition_threshold
        self.reload()

    def reload(self):
        db = load_db()
        self.names = sorted(db.keys())
        self.matrix = (
            np.stack([db[n] for n in self.names], axis=0) if self.names else None
        )
        print(f"[Recognizer] loaded {len(self.names)} identities "
              f"(threshold dist={self.threshold:.2f})")

    def detect(self, frame_bgr: np.ndarray, max_faces: int = 5):
        return self.detector.detect(frame_bgr, max_faces=max_faces)

    def identify(self, frame_bgr: np.ndarray, face: FaceDet) -> Tuple[Optional[str], float, bool]:
        """
        Returns (name_or_None, cosine_distance, accepted).
        """
        aligned, _ = align_face_5pt(frame_bgr, face.kps, out_size=(112, 112))
        emb = self.embedder.embed(aligned)

        if self.matrix is None or not self.names:
            return None, 1.0, False

        sims = self.matrix @ emb.reshape(-1)
        i = int(np.argmax(sims))
        sim = float(sims[i])
        dist = 1.0 - sim
        accepted = dist <= self.threshold
        return (self.names[i] if accepted else None), dist, accepted


# ---------------- standalone demo ----------------
def main():
    rec = Recognizer()
    cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Camera not available.")

    print("[recognize] q=quit | r=reload DB | +/- threshold")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if cfg.mirror_camera:
            frame = cv2.flip(frame, 1)

        vis = frame.copy()
        faces = rec.detect(frame, max_faces=5)

        for f in faces:
            name, dist, accepted = rec.identify(frame, f)
            color = (0, 255, 0) if accepted else (0, 0, 255)
            label = name if accepted else "Unknown"
            cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, 2)
            cv2.putText(vis, f"{label} d={dist:.2f}",
                        (f.x1, max(0, f.y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(vis, f"IDs={len(rec.names)} thr={rec.threshold:.2f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("recognize", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            rec.reload()
        elif key in (ord("+"), ord("=")):
            rec.threshold = min(1.20, rec.threshold + 0.01)
            print(f"[recognize] thr={rec.threshold:.2f}")
        elif key == ord("-"):
            rec.threshold = max(0.05, rec.threshold - 0.01)
            print(f"[recognize] thr={rec.threshold:.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()