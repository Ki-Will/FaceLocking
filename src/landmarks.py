import cv2
import numpy as np
import mediapipe as mp


IDX_LEFT_EYE = 33
IDX_RIGHT_EYE = 263
IDX_NOSE_TIP = 1
IDX_MOUTH_LEFT = 61
IDX_MOUTH_RIGHT = 291


def main():

    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    face = cv2.CascadeClassifier(
        cascade_path
    )

    if face.empty():
        raise RuntimeError(
            f"Failed to load cascade: {cascade_path}"
        )

    fm = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(
            "Camera not opened."
        )

    print(
        "Haar + FaceMesh 5pt. "
        "Press 'q' to quit."
    )

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        H, W = frame.shape[:2]

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        faces = face.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        for x, y, w, h in faces:
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        result = fm.process(rgb)

        if result.multi_face_landmarks:

            lm = (
                result
                .multi_face_landmarks[0]
                .landmark
            )

            indices = [
                IDX_LEFT_EYE,
                IDX_RIGHT_EYE,
                IDX_NOSE_TIP,
                IDX_MOUTH_LEFT,
                IDX_MOUTH_RIGHT,
            ]

            points = []

            for i in indices:

                p = lm[i]

                points.append(
                    [
                        p.x * W,
                        p.y * H,
                    ]
                )

            kps = np.array(
                points,
                dtype=np.float32,
            )

            if kps[0, 0] > kps[1, 0]:
                kps[[0, 1]] = kps[[1, 0]]

            if kps[3, 0] > kps[4, 0]:
                kps[[3, 4]] = kps[[4, 3]]

            for px, py in kps.astype(int):

                cv2.circle(
                    frame,
                    (int(px), int(py)),
                    4,
                    (0, 255, 0),
                    -1,
                )

        cv2.putText(
            frame,
            "5-point landmarks",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow(
            "5pt Landmarks",
            frame,
        )

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()