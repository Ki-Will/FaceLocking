# src/track.py
"""
Horizontal face tracking -> MQTT servo control.

Pipeline:
    camera frame
      -> YuNet detection (reuses src/detect_landmarks.py)
      -> pick target face (largest, weighted toward frame center)
      -> compute horizontal offset from frame center (normalized -1..+1)
      -> smooth offset (EMA)
      -> map to servo angle [SERVO_MIN..SERVO_MAX]
      -> publish angle via MqttBridge

Run:
    python -m src.track
Keys:
    q : quit
    c : re-center servo at SERVO_START_ANGLE
    +/- : widen / narrow dead-zone (pixels of tolerance around center)
    d : toggle debug overlay
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from .detect_landmarks import YuNetDetector, FaceDet
from .mqtt_bridge import MqttBridge


# ============================================================
# CONFIG  (mirrors ESP8266/main.py)
# ============================================================

SERVO_MIN_ANGLE = 10
SERVO_MAX_ANGLE = 170
SERVO_START_ANGLE = 90

# How aggressively we react. 1.0 = full sweep, 0.6 = gentler.
GAIN = 0.7

# Exponential moving average factor for the horizontal offset.
# Higher = smoother but slower. Range (0, 1).
SMOOTH_ALPHA = 0.75

# Dead-zone in pixels around the frame center. No movement inside it.
DEAD_ZONE_PX = 30

# Minimum interval between MQTT publishes (seconds).
MIN_PUBLISH_INTERVAL_S = 0.08

# Frame resize for detection speed (YuNet likes small inputs).
DETECT_WIDTH = 480


# ============================================================
# HELPERS
# ============================================================

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def pick_target(faces: List[FaceDet], frame_w: int) -> Optional[FaceDet]:
    """
    Pick the face to track:
    - Prefer the largest face.
    - Tie-break toward the frame center.
    """
    if not faces:
        return None

    cx_frame = frame_w / 2.0

    def score(f: FaceDet) -> float:
        w = f.x2 - f.x1
        h = f.y2 - f.y1
        area = w * h
        cx = (f.x1 + f.x2) / 2.0
        center_penalty = abs(cx - cx_frame) / cx_frame  # 0 = centered, 1 = edge
        return area * (1.0 - 0.35 * center_penalty)

    return max(faces, key=score)


@dataclass
class TrackerState:
    smooth_offset: float = 0.0     # normalized -1..+1
    last_angle: int = SERVO_START_ANGLE
    has_target: bool = False


# ============================================================
# MAIN
# ============================================================

def main():
    det = YuNetDetector(
        model_path="models/face_detection_yunet_2023mar.onnx",
        input_size=(320, 320),
        score_thresh=0.85,
        nms_thresh=0.3,
    )

    bridge = MqttBridge(
        broker="broker.hivemq.com",
        port=1883,
        client_id="pc-face-tracker",
        servo_topic="face/servo/angle",
        status_topic="face/tracking/status",
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not available.")

    print("[track] Connecting to MQTT...")
    bridge.connect()

    # Send the start angle once so the servo snaps to a known pose.
    bridge.publish_angle(SERVO_START_ANGLE, force=True)

    state = TrackerState(last_angle=SERVO_START_ANGLE)

    show_debug = True
    dead_zone_px = DEAD_ZONE_PX

    print("[track] Running. q=quit | c=recenter | +/- dead-zone | d=debug")

    # FPS bookkeeping
    t0 = time.time()
    frames = 0
    fps: Optional[float] = None

    # Status publish throttle
    last_status = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Flip so the view is mirror-like (natural for the operator).
            frame = cv2.flip(frame, 1)
            h_full, w_full = frame.shape[:2]

            # Downscale for detection speed.
            scale = DETECT_WIDTH / float(w_full)
            small = cv2.resize(frame, (DETECT_WIDTH, int(h_full * scale)))

            faces = det.detect(small, max_faces=5)

            # Scale boxes/keypoints back to full-frame coords.
            inv = 1.0 / scale
            for f in faces:
                f.x1 = int(f.x1 * inv); f.y1 = int(f.y1 * inv)
                f.x2 = int(f.x2 * inv); f.y2 = int(f.y2 * inv)
                f.kps = f.kps * inv

            target = pick_target(faces, w_full)

            vis = frame.copy()
            cx_frame = w_full / 2.0

            # Draw center line + dead-zone
            cv2.line(vis, (int(cx_frame), 0), (int(cx_frame), h_full), (255, 255, 0), 1)
            cv2.line(
                vis,
                (int(cx_frame - dead_zone_px), 0),
                (int(cx_frame - dead_zone_px), h_full),
                (128, 128, 0), 1,
            )
            cv2.line(
                vis,
                (int(cx_frame + dead_zone_px), 0),
                (int(cx_frame + dead_zone_px), h_full),
                (128, 128, 0), 1,
            )

            if target is not None:
                state.has_target = True

                cx = (target.x1 + target.x2) / 2.0
                offset_px = cx - cx_frame

                # Normalize to -1..+1 based on half-frame width.
                offset_norm = offset_px / cx_frame
                offset_norm = clamp(offset_norm, -1.0, 1.0)

                # Smooth
                state.smooth_offset = (
                    SMOOTH_ALPHA * state.smooth_offset
                    + (1.0 - SMOOTH_ALPHA) * offset_norm
                )

                # Map to angle. Positive offset -> face on right ->
                # turn servo toward the right.
                mid = (SERVO_MIN_ANGLE + SERVO_MAX_ANGLE) / 2.0
                span = (SERVO_MAX_ANGLE - SERVO_MIN_ANGLE) / 2.0
                raw_angle = mid + GAIN * span * state.smooth_offset
                angle = int(round(clamp(raw_angle, SERVO_MIN_ANGLE, SERVO_MAX_ANGLE)))

                # Dead-zone check on the raw pixel offset.
                if abs(offset_px) < dead_zone_px:
                    angle = state.last_angle  # hold
                else:
                    state.last_angle = angle

                # Draw face box + keypoints
                cv2.rectangle(vis, (target.x1, target.y1), (target.x2, target.y2), (0, 255, 0), 2)
                for (x, y) in target.kps.astype(int):
                    cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)

                cv2.putText(
                    vis, f"angle={angle}", (target.x1, max(0, target.y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )

                # Publish
                bridge.publish_angle(
                    angle,
                    force=False,
                    min_interval_s=MIN_PUBLISH_INTERVAL_S,
                )

                # Optional status publish (throttled to 2 Hz)
                now = time.time()
                if now - last_status > 0.5:
                    bridge.publish_status({
                        "has_target": True,
                        "offset_px": int(offset_px),
                        "offset_norm": round(offset_norm, 3),
                        "angle": angle,
                    })
                    last_status = now
            else:
                state.has_target = False

                # No face: fade back toward center.
                state.smooth_offset *= 0.9
                mid = (SERVO_MIN_ANGLE + SERVO_MAX_ANGLE) / 2.0
                span = (SERVO_MAX_ANGLE - SERVO_MIN_ANGLE) / 2.0
                angle = int(round(clamp(
                    mid + GAIN * span * state.smooth_offset,
                    SERVO_MIN_ANGLE, SERVO_MAX_ANGLE,
                )))
                state.last_angle = angle
                bridge.publish_angle(angle, min_interval_s=0.15)

                cv2.putText(
                    vis, "no face", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
                )

                now = time.time()
                if now - last_status > 0.5:
                    bridge.publish_status({"has_target": False, "angle": angle})
                    last_status = now

            # Debug overlay
            if show_debug:
                frames += 1
                dt = time.time() - t0
                if dt >= 1.0:
                    fps = frames / dt
                    frames = 0
                    t0 = time.time()

                info = (
                    f"faces={len(faces)} "
                    f"target={'Y' if state.has_target else 'N'} "
                    f"offset={state.smooth_offset:+.2f} "
                    f"angle={state.last_angle} "
                    f"thr={bridge._last_angle} "
                    f"fps={fps:.1f}" if fps else
                    f"faces={len(faces)} target={'Y' if state.has_target else 'N'}"
                )
                cv2.putText(vis, info, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("face track", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                state.last_angle = SERVO_START_ANGLE
                bridge.publish_angle(SERVO_START_ANGLE, force=True)
                print(f"[track] recentered to {SERVO_START_ANGLE}")
            elif key in (ord("+"), ord("=")):
                dead_zone_px = min(200, dead_zone_px + 10)
                print(f"[track] dead-zone = {dead_zone_px}px")
            elif key == ord("-"):
                dead_zone_px = max(0, dead_zone_px - 10)
                print(f"[track] dead-zone = {dead_zone_px}px")
            elif key == ord("d"):
                show_debug = not show_debug
                print(f"[track] debug = {'ON' if show_debug else 'OFF'}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        bridge.disconnect()


if __name__ == "__main__":
    main()