from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .haar_5pt import (
    Haar5ptDetector,
    align_face_5pt,
)


def main():

    cap = cv2.VideoCapture(0)

    detector = Haar5ptDetector(
        min_size=(70, 70),
        smooth_alpha=0.80,
        debug=False,
    )

    save_dir = Path(
        "data/debug_aligned"
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    blank = np.zeros(
        (
            112,
            112,
            3,
        ),
        dtype=np.uint8,
    )

    last_aligned = blank.copy()

    print(
        "Alignment running."
    )

    print(
        "q = quit"
    )

    print(
        "s = save aligned face"
    )

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

            last_aligned = (
                aligned
            )

            cv2.imshow(
                "aligned_112",
                last_aligned,
            )

        else:

            cv2.imshow(
                "aligned_112",
                last_aligned,
            )

        cv2.putText(
            vis,
            "5pt -> 112x112",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.imshow(
            "Alignment",
            vis,
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):
            break

        if key == ord("s"):

            filename = (
                save_dir
                / f"{int(time.time() * 1000)}.jpg"
            )

            cv2.imwrite(
                str(filename),
                last_aligned,
            )

            print(
                f"Saved: {filename}"
            )

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()