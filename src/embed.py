from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
import onnxruntime as ort

from .haar_5pt import (
    Haar5ptDetector,
    align_face_5pt,
)


@dataclass
class EmbeddingResult:

    embedding: np.ndarray
    norm_before: float
    dim: int


class ArcFaceEmbedderONNX:

    def __init__(
        self,
        model_path: str = (
            "models/embedder_arcface.onnx"
        ),
        input_size: Tuple[
            int,
            int,
        ] = (
            112,
            112,
        ),
        debug: bool = False,
    ):

        self.in_w = int(
            input_size[0]
        )

        self.in_h = int(
            input_size[1]
        )

        self.debug = debug

        self.sess = (
            ort.InferenceSession(
                model_path,
                providers=[
                    "CPUExecutionProvider"
                ],
            )
        )

        self.in_name = (
            self.sess
            .get_inputs()[0]
            .name
        )

        self.out_name = (
            self.sess
            .get_outputs()[0]
            .name
        )

        if debug:

            print(
                "[embed] model:",
                model_path,
            )

            print(
                "[embed] input:",
                self.sess
                .get_inputs()[0]
                .shape,
            )

            print(
                "[embed] output:",
                self.sess
                .get_outputs()[0]
                .shape,
            )

    def _preprocess(
        self,
        image: np.ndarray,
    ) -> np.ndarray:

        if image.shape[:2] != (
            self.in_h,
            self.in_w,
        ):

            image = cv2.resize(
                image,
                (
                    self.in_w,
                    self.in_h,
                ),
            )

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        ).astype(
            np.float32
        )

        rgb = (
            rgb - 127.5
        ) / 128.0

        x = np.transpose(
            rgb,
            (
                2,
                0,
                1,
            ),
        )[None, ...]

        return x.astype(
            np.float32
        )

    @staticmethod
    def _l2_normalize(
        vector: np.ndarray,
    ):

        vector = (
            vector
            .astype(
                np.float32
            )
            .reshape(-1)
        )

        norm = float(
            np.linalg.norm(vector)
        )

        normalized = (
            vector
            / (norm + 1e-12)
        )

        return (
            normalized.astype(
                np.float32
            ),
            norm,
        )

    def embed(
        self,
        aligned_bgr: np.ndarray,
    ) -> EmbeddingResult:

        x = self._preprocess(
            aligned_bgr
        )

        output = self.sess.run(
            [self.out_name],
            {
                self.in_name: x
            },
        )[0]

        vector = (
            np.asarray(
                output,
                dtype=np.float32,
            )
            .reshape(-1)
        )

        normalized, norm = (
            self._l2_normalize(
                vector
            )
        )

        return EmbeddingResult(
            embedding=normalized,
            norm_before=norm,
            dim=normalized.size,
        )


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


def main():

    cap = cv2.VideoCapture(0)

    detector = Haar5ptDetector(
        min_size=(70, 70),
        smooth_alpha=0.80,
        debug=False,
    )

    embedder = (
        ArcFaceEmbedderONNX(
            debug=True
        )
    )

    previous = None

    print(
        "Embedding test."
    )

    print(
        "q = quit"
    )

    print(
        "p = print embedding"
    )

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        faces = detector.detect(
            frame,
            max_faces=1,
        )

        vis = frame.copy()

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

            result = (
                embedder.embed(
                    aligned
                )
            )

            text = (
                f"dim={result.dim} "
                f"norm={result.norm_before:.2f}"
            )

            cv2.putText(
                vis,
                text,
                (
                    10,
                    30,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            if previous is not None:

                similarity = (
                    cosine_similarity(
                        previous,
                        result.embedding,
                    )
                )

                cv2.putText(
                    vis,
                    f"cos(prev)={similarity:.3f}",
                    (
                        10,
                        60,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            previous = (
                result.embedding
            )

        cv2.imshow(
            "Face Embedding",
            vis,
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):
            break

        if (
            key == ord("p")
            and previous is not None
        ):

            print(
                "Embedding dimension:",
                previous.size,
            )

            print(
                "Min:",
                previous.min(),
            )

            print(
                "Max:",
                previous.max(),
            )

            print(
                "First 10:",
                previous[:10],
            )

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()