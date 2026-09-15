"""
Threshold evaluation using enrollment crops (aligned 112x112).
Run: python -m src.evaluate
"""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from .embed import ArcFaceEmbedderONNX


ENROLL_DIR = Path("data/enroll")
MIN_IMGS = 5
MAX_IMGS = 80
TARGET_FAR = 0.01


def cosine_distance(a, b):
    return 1.0 - float(np.dot(a.reshape(-1), b.reshape(-1)))


def main():
    embedder = ArcFaceEmbedderONNX()

    people = sorted([p for p in ENROLL_DIR.iterdir() if p.is_dir()])
    per_person = {}
    for p in people:
        embs = []
        for img_path in sorted(p.glob("*.jpg"))[:MAX_IMGS]:
            img = cv2.imread(str(img_path))
            if img is None or img.shape[:2] != (112, 112):
                continue
            embs.append(embedder.embed(img))
        if len(embs) >= MIN_IMGS:
            per_person[p.name] = embs
        else:
            print(f"Skipping {p.name}: only {len(embs)} valid crops.")

    names = sorted(per_person.keys())
    if len(names) < 2:
        print("Enroll at least 2 people for meaningful evaluation.")
        return

    genuine, impostor = [], []
    for n in names:
        e = per_person[n]
        for i in range(len(e)):
            for j in range(i + 1, len(e)):
                genuine.append(cosine_distance(e[i], e[j]))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for a in per_person[names[i]]:
                for b in per_person[names[j]]:
                    impostor.append(cosine_distance(a, b))

    genuine = np.array(genuine, dtype=np.float32)
    impostor = np.array(impostor, dtype=np.float32)

    print(f"Genuine:  n={genuine.size} mean={genuine.mean():.3f} std={genuine.std():.3f}")
    print(f"Impostor: n={impostor.size} mean={impostor.mean():.3f} std={impostor.std():.3f}")

    best = None
    for thr in np.arange(0.10, 1.20, 0.01):
        far = float(np.mean(impostor <= thr))
        frr = float(np.mean(genuine > thr))
        if far <= TARGET_FAR:
            if best is None or frr < best[2]:
                best = (float(thr), far, frr)

    if best:
        thr, far, frr = best
        print(f"Suggested threshold: dist={thr:.2f} "
              f"(sim~{1.0-thr:.2f}) FAR={far*100:.2f}% FRR={frr*100:.2f}%")
    else:
        print("No threshold met the target FAR. Collect more varied samples.")


if __name__ == "__main__":
    main()