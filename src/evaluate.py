from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from .embed import (
    ArcFaceEmbedderONNX
)


ENROLL_DIR = Path(
    "data/enroll"
)

MIN_IMAGES = 3


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    return float(
        np.dot(
            a.reshape(-1),
            b.reshape(-1),
        )
    )


def cosine_distance(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    return 1.0 - cosine_similarity(
        a,
        b,
    )


def main():

    if not ENROLL_DIR.exists():

        print(
            "No enrollment directory."
        )

        return

    embedder = (
        ArcFaceEmbedderONNX(
            model_path=(
                "models/embedder_arcface.onnx"
            )
        )
    )

    people = {}

    for person_dir in sorted(
        ENROLL_DIR.iterdir()
    ):

        if not person_dir.is_dir():
            continue

        embeddings = []

        for path in sorted(
            person_dir.glob("*.jpg")
        ):

            image = cv2.imread(
                str(path)
            )

            if image is None:
                continue

            if image.shape[:2] != (
                112,
                112,
            ):
                continue

            result = (
                embedder.embed(
                    image
                )
            )

            embeddings.append(
                result.embedding
            )

        if len(
            embeddings
        ) >= MIN_IMAGES:

            people[
                person_dir.name
            ] = embeddings

    if not people:

        print(
            "Not enough enrollment data."
        )

        return

    genuine = []
    impostor = []

    names = sorted(
        people.keys()
    )

    for name in names:

        embs = people[name]

        for i in range(
            len(embs)
        ):

            for j in range(
                i + 1,
                len(embs),
            ):

                genuine.append(
                    cosine_distance(
                        embs[i],
                        embs[j],
                    )
                )

    for i in range(
        len(names)
    ):

        for j in range(
            i + 1,
            len(names),
        ):

            for a in people[
                names[i]
            ]:

                for b in people[
                    names[j]
                ]:

                    impostor.append(
                        cosine_distance(
                            a,
                            b,
                        )
                    )

    genuine = np.array(
        genuine,
        dtype=np.float32,
    )

    impostor = np.array(
        impostor,
        dtype=np.float32,
    )

    print()
    print(
        "=== FACE DATABASE EVALUATION ==="
    )

    if genuine.size:

        print(
            "Genuine:"
        )

        print(
            f"mean={genuine.mean():.3f} "
            f"std={genuine.std():.3f} "
            f"p95={np.percentile(genuine, 95):.3f}"
        )

    if impostor.size:

        print(
            "Impostor:"
        )

        print(
            f"mean={impostor.mean():.3f} "
            f"std={impostor.std():.3f} "
            f"p05={np.percentile(impostor, 5):.3f}"
        )

    print()

    best = None

    for threshold in np.arange(
        0.10,
        1.01,
        0.01,
    ):

        far = (
            np.mean(
                impostor
                <= threshold
            )
            if impostor.size
            else 0
        )

        frr = (
            np.mean(
                genuine
                > threshold
            )
            if genuine.size
            else 0
        )

        if far <= 0.01:

            candidate = (
                threshold,
                far,
                frr,
            )

            if (
                best is None
                or candidate[2]
                < best[2]
            ):

                best = candidate

    if best:

        threshold, far, frr = best

        print(
            f"Suggested distance threshold: "
            f"{threshold:.2f}"
        )

        print(
            f"FAR: {far * 100:.2f}%"
        )

        print(
            f"FRR: {frr * 100:.2f}%"
        )

    else:

        print(
            "No threshold achieved "
            "the requested FAR."
        )


if __name__ == "__main__":
    main()