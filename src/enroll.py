# src/enroll.py  (only the changed portions shown; the rest is unchanged)

# ...inside main():

"""
Enrollment tool: capture aligned faces, compute embeddings, store template.
Controls: SPACE=capture | a=auto | s=save | r=reset new | q=quit
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np

from .config import cfg
from .detect_landmarks import YuNetDetector
from .align import align_face_5pt
from .embed import ArcFaceEmbedderONNX

DB_NPZ = Path("data/db/face_db.npz")
DB_JSON = Path("data/db/face_db.json")
CROPS_DIR = Path("data/enroll")
SAMPLES_NEEDED = 15


def mean_embedding(embs: List[np.ndarray]) -> np.ndarray:
    E = np.stack([e.reshape(-1) for e in embs], axis=0).astype(np.float32)
    m = E.mean(axis=0)
    return (m / (np.linalg.norm(m) + 1e-12)).astype(np.float32)


def load_db() -> Dict[str, np.ndarray]:
    if DB_NPZ.exists():
        data = np.load(DB_NPZ, allow_pickle=True)
        return {k: data[k].astype(np.float32) for k in data.files}
    return {}


def save_db(db: Dict[str, np.ndarray], meta: dict) -> None:
    DB_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(DB_NPZ, **{k: v.astype(np.float32) for k, v in db.items()})
    DB_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main():
    name = input("Enter person name to enroll: ").strip()
    if not name:
        print("No name provided. Exiting.")
        return

    det = YuNetDetector()
    emb = ArcFaceEmbedderONNX(model_path=cfg.arc_face_model)
    db = load_db()

    person_dir = CROPS_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)

    # Load any existing aligned crops for this person
    base_samples: List[np.ndarray] = []
    for p in sorted(person_dir.glob("*.jpg")):
        img = cv2.imread(str(p))
        if img is not None and img.shape[:2] == (112, 112):
            base_samples.append(emb.embed(img))

    new_samples: List[np.ndarray] = []
    auto = False
    last_auto = 0.0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not available.")

    print("Controls: SPACE=capture | a=auto | s=save | r=reset new | q=quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vis = frame.copy()
        faces = det.detect(frame, max_faces=1)
        aligned = None

        if faces:
            f = faces[0]
            cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 2)
            for (x, y) in f.kps.astype(int):
                cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
            aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))

        if aligned is not None:
            cv2.imshow("aligned_112", aligned)
        else:
            cv2.imshow("aligned_112", np.zeros((112, 112, 3), dtype=np.uint8))

        # Auto-capture
        now = time.time()
        if auto and aligned is not None and (now - last_auto) >= 0.25:
            new_samples.append(emb.embed(aligned))
            cv2.imwrite(str(person_dir / f"{int(now*1000)}.jpg"), aligned)
            last_auto = now

        total = len(base_samples) + len(new_samples)
        cv2.putText(vis, f"ENROLL: {name}  New: {len(new_samples)}  Total: {total}/{SAMPLES_NEEDED}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, f"Auto: {'ON' if auto else 'OFF'}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("enroll", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("a"):
            auto = not auto
        elif key == ord("r"):
            new_samples.clear()
        elif key == ord(" "):
            if aligned is not None:
                new_samples.append(emb.embed(aligned))
                cv2.imwrite(str(person_dir / f"{int(time.time()*1000)}.jpg"), aligned)
        elif key == ord("s"):
            all_samples = base_samples + new_samples
            if len(all_samples) < 3:
                print(f"Not enough samples (have {len(all_samples)}).")
                continue
            template = mean_embedding(all_samples)
            db[name] = template
            meta = {
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "embedding_dim": int(template.size),
                "names": sorted(db.keys()),
                "samples_used": int(len(all_samples)),
            }
            save_db(db, meta)
            print(f"Saved '{name}'. Total identities: {len(db)}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()