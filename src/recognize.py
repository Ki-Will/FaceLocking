# src/recognize.py

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

try:
    import mediapipe as mp
except Exception as e:
    mp = None
    _MP_IMPORT_ERROR = e

from .haar_5pt import align_face_5pt
from .tracker import HorizontalTracker
from .servo_controller import ServoController


# ============================================================
# CONFIG
# ============================================================

DB_PATH = Path("data/db/face_db.npz")
MODEL_PATH = Path(
    os.getenv(
        "ARC_FACE_MODEL",
        "models/embedder_arcface.onnx",
    )
)

CAMERA_INDEX = int(
    os.getenv("CAMERA_INDEX", "0")
)

RECOGNITION_THRESHOLD = float(
    os.getenv(
        "RECOGNITION_THRESHOLD",
        "0.34",
    )
)

RECOGNITION_EVERY_N_FRAMES = int(
    os.getenv(
        "RECOGNITION_EVERY_N_FRAMES",
        "5",
    )
)

TARGET_CONFIRMATION_FRAMES = int(
    os.getenv(
        "TARGET_CONFIRMATION_FRAMES",
        "3",
    )
)

TARGET_LOST_FRAMES = int(
    os.getenv(
        "TARGET_LOST_FRAMES",
        "15",
    )
)

MIRROR_CAMERA = (
    os.getenv(
        "MIRROR_CAMERA",
        "true",
    ).lower()
    == "true"
)


# ============================================================
# MQTT CONFIG
# ============================================================

MQTT_BROKER = os.getenv(
    "MQTT_BROKER",
    "broker.hivemq.com",
)

MQTT_PORT = int(
    os.getenv(
        "MQTT_PORT",
        "1883",
    )
)

MQTT_SERVO_TOPIC = os.getenv(
    "MQTT_SERVO_TOPIC",
    "face/servo/angle",
)

MQTT_STATUS_TOPIC = os.getenv(
    "MQTT_STATUS_TOPIC",
    "face/tracking/status",
)

MQTT_DATA_TOPIC = os.getenv(
    "MQTT_DATA_TOPIC",
    "face/tracking/data",
)


# ============================================================
# TRACKING CONFIG
# ============================================================

TRACKER = HorizontalTracker(
    center_angle=int(
        os.getenv(
            "SERVO_CENTER_ANGLE",
            "90",
        )
    ),
    min_angle=int(
        os.getenv(
            "SERVO_MIN_ANGLE",
            "10",
        )
    ),
    max_angle=int(
        os.getenv(
            "SERVO_MAX_ANGLE",
            "170",
        )
    ),
    dead_zone=int(
        os.getenv(
            "TRACKING_DEAD_ZONE",
            "35",
        )
    ),
    kp=float(
        os.getenv(
            "TRACKING_KP",
            "0.08",
        )
    ),
    max_step=int(
        os.getenv(
            "TRACKING_MAX_STEP",
            "4",
        )
    ),
    min_change=int(
        os.getenv(
            "SERVO_MIN_CHANGE",
            "1",
        )
    ),
)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class FaceDet:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray


@dataclass
class MatchResult:
    name: Optional[str]
    distance: float
    similarity: float
    accepted: bool


# ============================================================
# GEOMETRY
# ============================================================

def _clip_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    W: int,
    H: int,
) -> Tuple[int, int, int, int]:

    x1 = int(
        max(
            0,
            min(
                W - 1,
                round(x1),
            ),
        )
    )

    y1 = int(
        max(
            0,
            min(
                H - 1,
                round(y1),
            ),
        )
    )

    x2 = int(
        max(
            0,
            min(
                W - 1,
                round(x2),
            ),
        )
    )

    y2 = int(
        max(
            0,
            min(
                H - 1,
                round(y2),
            ),
        )
    )

    if x2 < x1:
        x1, x2 = x2, x1

    if y2 < y1:
        y1, y2 = y2, y1

    return x1, y1, x2, y2


def _bbox_from_5pt(
    kps: np.ndarray,
    pad_x: float = 0.55,
    pad_y_top: float = 0.85,
    pad_y_bot: float = 1.15,
) -> np.ndarray:

    k = kps.astype(np.float32)

    x_min = float(np.min(k[:, 0]))
    x_max = float(np.max(k[:, 0]))

    y_min = float(np.min(k[:, 1]))
    y_max = float(np.max(k[:, 1]))

    w = max(
        1.0,
        x_max - x_min,
    )

    h = max(
        1.0,
        y_max - y_min,
    )

    x1 = x_min - pad_x * w
    x2 = x_max + pad_x * w

    y1 = y_min - pad_y_top * h
    y2 = y_max + pad_y_bot * h

    return np.array(
        [
            x1,
            y1,
            x2,
            y2,
        ],
        dtype=np.float32,
    )


def _kps_span_ok(
    kps: np.ndarray,
    min_eye_dist: float,
) -> bool:

    k = kps.astype(np.float32)

    le, re, no, lm, rm = k

    eye_dist = float(
        np.linalg.norm(
            re - le
        )
    )

    if eye_dist < float(min_eye_dist):
        return False

    if not (
        lm[1] > no[1]
        and rm[1] > no[1]
    ):
        return False

    return True


# ============================================================
# MATH
# ============================================================

def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    a = a.reshape(-1).astype(
        np.float32
    )

    b = b.reshape(-1).astype(
        np.float32
    )

    return float(
        np.dot(a, b)
    )


# ============================================================
# DATABASE
# ============================================================

def load_db_npz(
    db_path: Path,
) -> Dict[str, np.ndarray]:

    if not db_path.exists():
        return {}

    data = np.load(
        str(db_path),
        allow_pickle=True,
    )

    out: Dict[
        str,
        np.ndarray,
    ] = {}

    for key in data.files:
        out[key] = (
            np.asarray(
                data[key],
                dtype=np.float32,
            )
            .reshape(-1)
        )

    return out


# ============================================================
# ARCFACE EMBEDDER
# ============================================================

class ArcFaceEmbedderONNX:

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[
            int,
            int,
        ] = (112, 112),
    ):

        self.in_w = int(
            input_size[0]
        )

        self.in_h = int(
            input_size[1]
        )

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

    def _preprocess(
        self,
        image: np.ndarray,
    ) -> np.ndarray:

        if (
            image.shape[1]
            != self.in_w
            or image.shape[0]
            != self.in_h
        ):
            image = cv2.resize(
                image,
                (
                    self.in_w,
                    self.in_h,
                ),
                interpolation=cv2.INTER_LINEAR,
            )

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        ).astype(np.float32)

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
        v: np.ndarray,
    ) -> np.ndarray:

        v = (
            v.astype(
                np.float32
            )
            .reshape(-1)
        )

        norm = float(
            np.linalg.norm(v)
            + 1e-12
        )

        return (
            v / norm
        ).astype(
            np.float32
        )

    def embed(
        self,
        image: np.ndarray,
    ) -> np.ndarray:

        x = self._preprocess(
            image
        )

        y = self.sess.run(
            [self.out_name],
            {
                self.in_name: x
            },
        )[0]

        emb = np.asarray(
            y,
            dtype=np.float32,
        ).reshape(-1)

        return self._l2_normalize(
            emb
        )


# ============================================================
# FACE DETECTOR
# ============================================================

class HaarFaceMesh5pt:

    def __init__(
        self,
        min_size: Tuple[
            int,
            int,
        ] = (70, 70),
    ):

        self.min_size = tuple(
            map(
                int,
                min_size,
            )
        )

        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        self.face_cascade = (
            cv2.CascadeClassifier(
                cascade_path
            )
        )

        if self.face_cascade.empty():
            raise RuntimeError(
                "Failed to load Haar cascade."
            )

        if mp is None:
            raise RuntimeError(
                "MediaPipe import failed:\n"
                f"{_MP_IMPORT_ERROR}\n\n"
                "Install with:\n"
                "pip install mediapipe==0.10.21"
            )

        self.mesh = (
            mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

        self.IDX_LEFT_EYE = 33
        self.IDX_RIGHT_EYE = 263
        self.IDX_NOSE_TIP = 1
        self.IDX_MOUTH_LEFT = 61
        self.IDX_MOUTH_RIGHT = 291

    def _haar_faces(
        self,
        gray: np.ndarray,
    ) -> np.ndarray:

        faces = (
            self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                flags=cv2.CASCADE_SCALE_IMAGE,
                minSize=self.min_size,
            )
        )

        if (
            faces is None
            or len(faces) == 0
        ):
            return np.zeros(
                (0, 4),
                dtype=np.int32,
            )

        return faces.astype(
            np.int32
        )

    def _roi_facemesh_5pt(
        self,
        roi_bgr: np.ndarray,
    ) -> Optional[np.ndarray]:

        H, W = roi_bgr.shape[:2]

        if H < 20 or W < 20:
            return None

        rgb = cv2.cvtColor(
            roi_bgr,
            cv2.COLOR_BGR2RGB,
        )

        result = self.mesh.process(
            rgb
        )

        if not result.multi_face_landmarks:
            return None

        lm = (
            result
            .multi_face_landmarks[0]
            .landmark
        )

        indices = [
            self.IDX_LEFT_EYE,
            self.IDX_RIGHT_EYE,
            self.IDX_NOSE_TIP,
            self.IDX_MOUTH_LEFT,
            self.IDX_MOUTH_RIGHT,
        ]

        points = []

        for index in indices:
            point = lm[index]

            points.append(
                [
                    point.x * W,
                    point.y * H,
                ]
            )

        kps = np.array(
            points,
            dtype=np.float32,
        )

        if kps[0, 0] > kps[1, 0]:
            kps[[0, 1]] = (
                kps[[1, 0]]
            )

        if kps[3, 0] > kps[4, 0]:
            kps[[3, 4]] = (
                kps[[4, 3]]
            )

        return kps

    def detect(
        self,
        frame_bgr: np.ndarray,
        max_faces: int = 5,
    ) -> List[FaceDet]:

        H, W = frame_bgr.shape[:2]

        gray = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2GRAY,
        )

        faces = self._haar_faces(
            gray
        )

        if faces.shape[0] == 0:
            return []

        areas = (
            faces[:, 2]
            * faces[:, 3]
        )

        order = np.argsort(
            areas
        )[::-1]

        faces = faces[
            order
        ][:max_faces]

        output: List[
            FaceDet
        ] = []

        for (
            x,
            y,
            w,
            h,
        ) in faces:

            mx = 0.25 * w
            my = 0.35 * h

            rx1, ry1, rx2, ry2 = (
                _clip_xyxy(
                    x - mx,
                    y - my,
                    x + w + mx,
                    y + h + my,
                    W,
                    H,
                )
            )

            roi = frame_bgr[
                ry1:ry2,
                rx1:rx2,
            ]

            kps_roi = (
                self._roi_facemesh_5pt(
                    roi
                )
            )

            if kps_roi is None:
                continue

            kps = kps_roi.copy()

            kps[:, 0] += float(
                rx1
            )

            kps[:, 1] += float(
                ry1
            )

            if not _kps_span_ok(
                kps,
                min_eye_dist=max(
                    10.0,
                    0.18 * float(w),
                ),
            ):
                continue

            bbox = (
                _bbox_from_5pt(
                    kps
                )
            )

            x1, y1, x2, y2 = (
                _clip_xyxy(
                    bbox[0],
                    bbox[1],
                    bbox[2],
                    bbox[3],
                    W,
                    H,
                )
            )

            output.append(
                FaceDet(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    score=1.0,
                    kps=kps.astype(
                        np.float32
                    ),
                )
            )

        return output


# ============================================================
# MATCHER
# ============================================================

class FaceDBMatcher:

    def __init__(
        self,
        db: Dict[
            str,
            np.ndarray,
        ],
        threshold: float,
    ):

        self.db = db
        self.threshold = float(
            threshold
        )

        self.names: List[
            str
        ] = []

        self.matrix: Optional[
            np.ndarray
        ] = None

        self.rebuild()

    def rebuild(self):

        self.names = sorted(
            self.db.keys()
        )

        if not self.names:
            self.matrix = None
            return

        self.matrix = np.stack(
            [
                self.db[name]
                .reshape(-1)
                .astype(
                    np.float32
                )
                for name in self.names
            ],
            axis=0,
        )

    def reload(
        self,
        path: Path,
    ):

        self.db = load_db_npz(
            path
        )

        self.rebuild()

    def match(
        self,
        embedding: np.ndarray,
    ) -> MatchResult:

        if (
            self.matrix is None
            or not self.names
        ):
            return MatchResult(
                name=None,
                distance=1.0,
                similarity=0.0,
                accepted=False,
            )

        emb = (
            embedding
            .reshape(1, -1)
            .astype(np.float32)
        )

        similarities = (
            self.matrix @ emb.T
        ).reshape(-1)

        best_index = int(
            np.argmax(
                similarities
            )
        )

        best_similarity = float(
            similarities[
                best_index
            ]
        )

        best_distance = (
            1.0
            - best_similarity
        )

        accepted = (
            best_distance
            <= self.threshold
        )

        return MatchResult(
            name=(
                self.names[
                    best_index
                ]
                if accepted
                else None
            ),
            distance=best_distance,
            similarity=best_similarity,
            accepted=accepted,
        )


# ============================================================
# UI
# ============================================================

def draw_text(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    scale: float = 0.7,
    color=(255, 255, 255),
    thickness: int = 2,
):

    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_center_line(
    image: np.ndarray,
):

    h, w = image.shape[:2]

    center_x = w // 2

    cv2.line(
        image,
        (
            center_x,
            0,
        ),
        (
            center_x,
            h,
        ),
        (255, 255, 0),
        1,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print(" FACE RECOGNITION + HORIZONTAL SERVO TRACKING")
    print("=" * 60)
    print()

    # --------------------------------------------------------
    # Database
    # --------------------------------------------------------

    db = load_db_npz(
        DB_PATH
    )

    if not db:
        raise RuntimeError(
            "No face database found.\n"
            "Run enrollment first:\n"
            "python -m src.enroll"
        )

    print(
        f"[DB] Loaded {len(db)} identities:"
    )

    for name in sorted(
        db.keys()
    ):
        print(
            f"     - {name}"
        )

    # --------------------------------------------------------
    # Vision
    # --------------------------------------------------------

    detector = HaarFaceMesh5pt(
        min_size=(70, 70)
    )

    embedder = ArcFaceEmbedderONNX(
        model_path=str(
            MODEL_PATH
        ),
        input_size=(112, 112),
    )

    matcher = FaceDBMatcher(
        db=db,
        threshold=RECOGNITION_THRESHOLD,
    )

    # --------------------------------------------------------
    # MQTT
    # --------------------------------------------------------

    servo = ServoController(
        broker=MQTT_BROKER,
        port=MQTT_PORT,
        servo_topic=MQTT_SERVO_TOPIC,
        status_topic=MQTT_STATUS_TOPIC,
        data_topic=MQTT_DATA_TOPIC,
    )

    try:
        servo.connect()
    except Exception as exc:
        print(
            f"[MQTT] WARNING: {exc}"
        )
        print(
            "[MQTT] Recognition will continue, "
            "but servo tracking will not work."
        )

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not cap.isOpened():
        servo.disconnect()
        raise RuntimeError(
            f"Camera {CAMERA_INDEX} "
            "could not be opened."
        )

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    frame_number = 0

    current_target: Optional[
        str
    ] = None

    candidate_target: Optional[
        str
    ] = None

    candidate_count = 0

    lost_frames = 0

    last_recognition: List[
        MatchResult
    ] = []

    last_recognition_faces: List[
        FaceDet
    ] = []

    last_recognition_time = 0.0

    status = "SEARCHING"

    fps_start = time.time()
    fps_frames = 0
    fps = 0.0

    # --------------------------------------------------------
    # Initial servo center
    # --------------------------------------------------------

    center_angle = TRACKER.reset()

    if servo.connected:
        servo.publish_angle(
            center_angle
        )

        servo.publish_status(
            "SEARCHING"
        )

    print()
    print(
        "System running."
    )
    print(
        "q = quit"
    )
    print(
        "r = reload database"
    )
    print(
        "+ / - = recognition threshold"
    )
    print(
        "c = center servo"
    )
    print()

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                print(
                    "[CAMERA] Failed to read frame."
                )
                break

            if MIRROR_CAMERA:
                frame = cv2.flip(
                    frame,
                    1,
                )

            vis = frame.copy()

            H, W = frame.shape[:2]

            draw_center_line(
                vis
            )

            # ------------------------------------------------
            # Detection
            # ------------------------------------------------

            faces = detector.detect(
                frame,
                max_faces=5,
            )

            # ------------------------------------------------
            # Recognition
            # ------------------------------------------------

            should_recognize = (
                frame_number
                % max(
                    1,
                    RECOGNITION_EVERY_N_FRAMES,
                )
                == 0
            )

            if should_recognize:

                last_recognition = []
                last_recognition_faces = []

                for face in faces:

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

                    embedding = (
                        embedder.embed(
                            aligned
                        )
                    )

                    match = matcher.match(
                        embedding
                    )

                    last_recognition.append(
                        match
                    )

                    last_recognition_faces.append(
                        face
                    )

                last_recognition_time = (
                    time.time()
                )

            # ------------------------------------------------
            # Associate recognition results with current faces
            # ------------------------------------------------

            recognized_target_face = None
            recognized_target_match = None

            # First use the latest recognition result.
            #
            # We look for the currently selected target name.
            if current_target is not None:

                for (
                    face,
                    match,
                ) in zip(
                    last_recognition_faces,
                    last_recognition,
                ):

                    if (
                        match.accepted
                        and match.name
                        == current_target
                    ):

                        recognized_target_face = (
                            face
                        )

                        recognized_target_match = (
                            match
                        )

                        break

            # If no target exists, look for a known person.
            if (
                current_target is None
                and last_recognition
            ):

                for (
                    face,
                    match,
                ) in zip(
                    last_recognition_faces,
                    last_recognition,
                ):

                    if (
                        match.accepted
                        and match.name
                    ):

                        if (
                            candidate_target
                            == match.name
                        ):
                            candidate_count += 1
                        else:
                            candidate_target = (
                                match.name
                            )
                            candidate_count = 1

                        # Confirmation prevents a single
                        # noisy recognition from starting
                        # tracking.
                        if (
                            candidate_count
                            >= TARGET_CONFIRMATION_FRAMES
                        ):

                            current_target = (
                                match.name
                            )

                            lost_frames = 0
                            status = (
                                "TRACKING"
                            )

                            print(
                                f"[TRACK] Target acquired: "
                                f"{current_target}"
                            )

                            if servo.connected:
                                servo.publish_status(
                                    "TRACKING"
                                )

                            recognized_target_face = (
                                face
                            )

                            recognized_target_match = (
                                match
                            )

                        break

            # ------------------------------------------------
            # Tracking
            # ------------------------------------------------

            if (
                current_target is not None
            ):

                if (
                    recognized_target_face
                    is not None
                ):

                    lost_frames = 0

                    face = (
                        recognized_target_face
                    )

                    match = (
                        recognized_target_match
                    )

                    face_center_x = (
                        (
                            face.x1
                            + face.x2
                        )
                        / 2.0
                    )

                    tracking = TRACKER.update(
                        face_x=face_center_x,
                        frame_width=W,
                    )

                    if (
                        tracking.moving
                        and servo.connected
                    ):

                        servo.publish_angle(
                            tracking.servo_angle
                        )

                    if servo.connected:

                        servo.publish_tracking_data(
                            name=current_target,
                            face_x=tracking.face_x,
                            frame_center_x=tracking.frame_center_x,
                            error_x=tracking.error_x,
                            servo_angle=tracking.servo_angle,
                            distance=match.distance,
                            similarity=match.similarity,
                        )

                else:

                    lost_frames += 1

                    if (
                        lost_frames
                        >= TARGET_LOST_FRAMES
                    ):

                        print(
                            f"[TRACK] Lost target: "
                            f"{current_target}"
                        )

                        current_target = None
                        candidate_target = None
                        candidate_count = 0

                        status = (
                            "SEARCHING"
                        )

                        if servo.connected:
                            servo.publish_status(
                                "SEARCHING"
                            )

            # ------------------------------------------------
            # Draw all detected faces
            # ------------------------------------------------

            for index, face in enumerate(
                faces
            ):

                x1 = face.x1
                y1 = face.y1
                x2 = face.x2
                y2 = face.y2

                # Determine recognition result
                match = None

                if index < len(
                    last_recognition
                ):
                    match = (
                        last_recognition[
                            index
                        ]
                    )

                if (
                    match is not None
                    and match.accepted
                    and match.name
                ):

                    label = match.name

                else:

                    label = "Unknown"

                # Highlight target.
                is_target = (
                    current_target
                    is not None
                    and label
                    == current_target
                )

                if is_target:
                    box_color = (
                        255,
                        0,
                        255,
                    )
                    thickness = 4
                elif (
                    match is not None
                    and match.accepted
                ):
                    box_color = (
                        0,
                        255,
                        0,
                    )
                    thickness = 2
                else:
                    box_color = (
                        0,
                        0,
                        255,
                    )
                    thickness = 2

                cv2.rectangle(
                    vis,
                    (
                        x1,
                        y1,
                    ),
                    (
                        x2,
                        y2,
                    ),
                    box_color,
                    thickness,
                )

                # 5 landmarks
                for (
                    px,
                    py,
                ) in face.kps.astype(
                    int
                ):

                    cv2.circle(
                        vis,
                        (
                            int(px),
                            int(py),
                        ),
                        3,
                        (
                            0,
                            255,
                            255,
                        ),
                        -1,
                    )

                # Face center
                cx = int(
                    (
                        x1 + x2
                    )
                    / 2
                )

                cy = int(
                    (
                        y1 + y2
                    )
                    / 2
                )

                cv2.circle(
                    vis,
                    (
                        cx,
                        cy,
                    ),
                    5,
                    (
                        255,
                        255,
                        0,
                    ),
                    -1,
                )

                # Label
                label_y = max(
                    25,
                    y1 - 10,
                )

                draw_text(
                    vis,
                    label,
                    (
                        x1,
                        label_y,
                    ),
                    0.7,
                    box_color,
                    2,
                )

                if match is not None:

                    draw_text(
                        vis,
                        (
                            f"dist="
                            f"{match.distance:.3f}"
                        ),
                        (
                            x1,
                            label_y + 23,
                        ),
                        0.55,
                        box_color,
                        2,
                    )

            # ------------------------------------------------
            # Tracking overlay
            # ------------------------------------------------

            if current_target is not None:

                # Find current target position
                target_face = None

                for (
                    face,
                    match,
                ) in zip(
                    last_recognition_faces,
                    last_recognition,
                ):

                    if (
                        match.accepted
                        and match.name
                        == current_target
                    ):

                        target_face = face
                        break

                if target_face is not None:

                    face_x = (
                        target_face.x1
                        + target_face.x2
                    ) / 2.0

                    error = (
                        face_x
                        - W / 2.0
                    )

                    cv2.line(
                        vis,
                        (
                            int(face_x),
                            0,
                        ),
                        (
                            int(face_x),
                            H,
                        ),
                        (
                            255,
                            0,
                            255,
                        ),
                        1,
                    )

                    draw_text(
                        vis,
                        (
                            f"TARGET: "
                            f"{current_target}"
                        ),
                        (
                            10,
                            65,
                        ),
                        0.75,
                        (
                            255,
                            0,
                            255,
                        ),
                        2,
                    )

                    draw_text(
                        vis,
                        (
                            f"X error: "
                            f"{error:+.0f}px"
                        ),
                        (
                            10,
                            92,
                        ),
                        0.65,
                        (
                            255,
                            255,
                            255,
                        ),
                        2,
                    )

                    draw_text(
                        vis,
                        (
                            f"Servo: "
                            f"{TRACKER.current_angle}°"
                        ),
                        (
                            10,
                            119,
                        ),
                        0.65,
                        (
                            255,
                            255,
                            255,
                        ),
                        2,
                    )

                else:

                    draw_text(
                        vis,
                        (
                            f"TARGET LOST: "
                            f"{current_target}"
                        ),
                        (
                            10,
                            65,
                        ),
                        0.75,
                        (
                            0,
                            165,
                            255,
                        ),
                        2,
                    )

            else:

                draw_text(
                    vis,
                    "SEARCHING FOR ENROLLED PERSON",
                    (
                        10,
                        65,
                    ),
                    0.7,
                    (
                        255,
                        255,
                        255,
                    ),
                    2,
                )

            # ------------------------------------------------
            # Header
            # ------------------------------------------------

            fps_frames += 1

            elapsed = (
                time.time()
                - fps_start
            )

            if elapsed >= 1.0:

                fps = (
                    fps_frames
                    / elapsed
                )

                fps_frames = 0
                fps_start = (
                    time.time()
                )

            draw_text(
                vis,
                (
                    f"STATUS: {status} | "
                    f"FPS: {fps:.1f} | "
                    f"IDs: {len(matcher.names)} | "
                    f"THR: "
                    f"{matcher.threshold:.2f}"
                ),
                (
                    10,
                    28,
                ),
                0.65,
                (
                    255,
                    255,
                    255,
                ),
                2,
            )

            # ------------------------------------------------
            # Display
            # ------------------------------------------------

            cv2.imshow(
                "Face Recognition + Tracking",
                vis,
            )

            frame_number += 1

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            # ------------------------------------------------
            # Controls
            # ------------------------------------------------

            if key == ord("q"):
                break

            elif key == ord("r"):

                matcher.reload(
                    DB_PATH
                )

                print(
                    "[DB] Reloaded: "
                    f"{len(matcher.names)} identities"
                )

            elif key in (
                ord("+"),
                ord("="),
            ):

                matcher.threshold = min(
                    1.20,
                    matcher.threshold
                    + 0.01,
                )

                print(
                    "[MATCH] threshold = "
                    f"{matcher.threshold:.2f}"
                )

            elif key == ord("-"):

                matcher.threshold = max(
                    0.05,
                    matcher.threshold
                    - 0.01,
                )

                print(
                    "[MATCH] threshold = "
                    f"{matcher.threshold:.2f}"
                )

            elif key == ord("c"):

                angle = TRACKER.reset()

                print(
                    f"[SERVO] Center -> "
                    f"{angle}°"
                )

                if servo.connected:
                    servo.publish_angle(
                        angle
                    )

            # ------------------------------------------------
            # Small status update
            # ------------------------------------------------

            if (
                current_target is None
                and candidate_target is not None
            ):

                status = (
                    "IDENTIFYING"
                )

            elif (
                current_target is not None
            ):

                status = (
                    "TRACKING"
                )

    finally:

        print()
        print(
            "[SYSTEM] Shutting down..."
        )

        cap.release()

        cv2.destroyAllWindows()

        if servo.connected:

            try:
                servo.publish_status(
                    "OFFLINE"
                )

                # Return camera to center
                servo.publish_angle(
                    TRACKER.center_angle
                )

            except Exception:
                pass

        servo.disconnect()

        print(
            "[SYSTEM] Shutdown complete."
        )


if __name__ == "__main__":
    main()