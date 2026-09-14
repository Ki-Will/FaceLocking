from __future__ import annotations

import json
import time

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from .haar_5pt import (
    Haar5ptDetector,
    align_face_5pt,
)

from .embed import (
    ArcFaceEmbedderONNX,
)


DB_NPZ = Path(
    "data/db/face_db.npz"
)

DB_JSON = Path(
    "data/db/face_db.json"
)

ENROLL_DIR = Path(
    "data/enroll"
)

SAMPLES_NEEDED = 15


def load_db() -> Dict[str, np.ndarray]:


    if not DB_NPZ.exists():
        return {}

    data = np.load(
        DB_NPZ,
        allow_pickle=True,
    )

    return {
        key: data[key].astype(
            np.float32
        )
        for key in data.files
    }


def save_db(
    db: Dict[str, np.ndarray],
):

    DB_NPZ.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        DB_NPZ,
        **db,
    )

    metadata = {
        "updated_at": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "embedding_dim": (
            int(
                next(
                    iter(
                        db.values()
                    )
                ).size
            )
            if db
            else 0
        ),
        "names": sorted(
            db.keys()
        ),
        "note": (
            "Embeddings are "
            "L2-normalized vectors. "
            "Matching uses cosine similarity."
        ),
    }

    DB_JSON.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )


def mean_embedding(
    embeddings: List[np.ndarray],
) -> np.ndarray:

    matrix = np.stack(
        embeddings,
        axis=0,
    ).astype(
        np.float32
    )

    mean = matrix.mean(
        axis=0
    )

    mean /= (
        np.linalg.norm(mean)
        + 1e-12
    )

    return mean.astype(
        np.float32
    )


def main():

    ENROLL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    name = input(
        "Enter person name: "
    ).strip()

    if not name:
        print(
            "No name entered."
        )
        return

    person_dir = (
        ENROLL_DIR / name
    )

    person_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    detector = Haar5ptDetector(
        min_size=(70, 70),
        smooth_alpha=0.80,
        debug=False,
    )

    embedder = (
        ArcFaceEmbedderONNX(
            model_path=(
                "models/embedder_arcface.onnx"
            )
        )
    )

    db = load_db()

    new_embeddings = []

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(
            "Camera could not be opened."
        )

    print()
    print(
        "Enrollment started."
    )
    print()
    print(
        "SPACE = capture"
    )
    print(
        "a = automatic capture"
    )
    print(
        "s = save"
    )
    print(
        "q = quit"
    )
    print()

    auto = False
    last_auto = 0

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            frame = cv2.flip(
                frame,
                1,
            )

            vis = frame.copy()

            faces = detector.detect(
                frame,
                max_faces=1,
            )

            aligned = None

            if faces:

                face = faces[0]

                cv2.rectangle(
                    vis,
                    (
                        face.x1,
                        face.y1,
                    ),
                    (
                        face.x2,
                        face.y2,
                    ),
                    (0, 255, 0),
                    2,
                )

                for px, py in (
                    face.kps.astype(int)
                ):

                    cv2.circle(
                        vis,
                        (
                            int(px),
                            int(py),
                        ),
                        3,
                        (0, 255, 0),
                        -1,
                    )

                aligned, _ = (
                    align_face_5pt(
                        frame,
                        face.kps,
                        out_size=(
                            112,
                            112,
                        ),
                    )
                )

                cv2.imshow(
                    "Aligned Face",
                    aligned,
                )

            else:

                cv2.imshow(
                    "Aligned Face",
                    np.zeros(
                        (
                            112,
                            112,
                            3,
                        ),
                        dtype=np.uint8,
                    ),
                )

            now = time.time()

            if (
                auto
                and aligned is not None
                and now - last_auto >= 0.25
            ):

                result = (
                    embedder.embed(
                        aligned
                    )
                )

                new_embeddings.append(
                    result.embedding
                )

                filename = (
                    person_dir
                    / f"{int(now * 1000)}.jpg"
                )

                cv2.imwrite(
                    str(filename),
                    aligned,
                )

                last_auto = now

            cv2.putText(
                vis,
                f"Person: {name}",
                (
                    10,
                    30,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                vis,
                (
                    f"Samples: "
                    f"{len(new_embeddings)}"
                    f"/{SAMPLES_NEEDED}"
                ),
                (
                    10,
                    60,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                vis,
                f"Auto: {'ON' if auto else 'OFF'}",
                (
                    10,
                    90,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "Enrollment",
                vis,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):
                break

            elif key == ord("a"):

                auto = not auto

                print(
                    "Auto:",
                    auto,
                )

            elif key == ord(" "):

                if aligned is None:

                    print(
                        "No face detected."
                    )

                    continue

                result = (
                    embedder.embed(
                        aligned
                    )
                )

                new_embeddings.append(
                    result.embedding
                )

                filename = (
                    person_dir
                    / f"{int(time.time() * 1000)}.jpg"
                )

                cv2.imwrite(
                    str(filename),
                    aligned,
                )

                print(
                    f"Captured "
                    f"{len(new_embeddings)}"
                    f"/{SAMPLES_NEEDED}"
                )

            elif key == ord("s"):

                if len(
                    new_embeddings
                ) < 3:

                    print(
                        "Capture at least "
                        "3 samples first."
                    )

                    continue

                db[name] = (
                    mean_embedding(
                        new_embeddings
                    )
                )

                save_db(
                    db
                )

                print()
                print(
                    f"Saved '{name}'."
                )

                print(
                    "Identities:",
                    sorted(
                        db.keys()
                    ),
                )

                new_embeddings.clear()

    finally:

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
