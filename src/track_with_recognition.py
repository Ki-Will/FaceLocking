# src/track_with_recognition.py
"""
Face tracking with identity gating.

Behavior:
  - Runs face detection on every frame (fast).
  - Runs recognition (align + embed + match) every N frames only.
  - A face becomes the TRACKED TARGET only after the same identity
    is seen for CONFIRMATION_FRAMES consecutive recognition cycles.
  - Once tracking, if the target disappears for up to LOST_FRAMES,
    we keep steering with the last known offset.
  - Servo is driven by proportional control (TRACKING_KP),
    clamped by TRACKING_MAX_STEP degrees per update, with a
    pixel dead-zone around frame center.

Run:
    python -m src.track_with_recognition
Keys:
    q : quit
    c : recenter servo
    r : reload DB
    d : toggle debug overlay
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .config import cfg
from .detect_landmarks import FaceDet
from .recognize import Recognizer
from .mqtt_bridge import MqttBridge


# ============================================================
# TRACK STATE
# ============================================================

@dataclass
class TrackState:
    target_name: Optional[str] = None       # currently confirmed identity
    confirmation_count: int = 0             # consecutive recognitions of same name
    pending_name: Optional[str] = None      # candidate awaiting confirmation
    lost_counter: int = 0                   # frames since target last seen
    last_offset_px: float = 0.0             # last known horizontal offset
    last_angle: int = cfg.servo_center_angle

    def reset(self):
        self.target_name = None
        self.confirmation_count = 0
        self.pending_name = None
        self.lost_counter = 0
        # keep last_offset_px and last_angle to avoid snap-back


# ============================================================
# HELPERS
# ============================================================

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def largest_face(faces) -> Optional[FaceDet]:
    if not faces:
        return None
    return max(faces, key=lambda f: (f.x2 - f.x1) * (f.y2 - f.y1))


def face_center_x(f: FaceDet) -> float:
    return (f.x1 + f.x2) / 2.0


def angle_from_offset(offset_px: float, frame_w: float) -> int:
    """
    Proportional mapping:
        offset_px normalized to [-1, +1] by half frame width
        scaled by KP and added to the center angle.
    """
    half = frame_w / 2.0
    norm = clamp(offset_px / half, -1.0, 1.0)
    center = cfg.servo_center_angle
    # KP is small (e.g. 0.08): 1.0 * half-frame width -> full half sweep * KP
    half_span = (cfg.servo_max_angle - cfg.servo_min_angle) / 2.0
    target = center + cfg.tracking_kp * half_span * norm
    return int(round(clamp(target, cfg.servo_min_angle, cfg.servo_max_angle)))


def slew_limit(current: int, target: int, max_step: int) -> int:
    delta = target - current
    if delta > max_step:
        delta = max_step
    elif delta < -max_step:
        delta = -max_step
    return int(current + delta)


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    print("[track] loading recognizer...")
    rec = Recognizer()
    bridge = MqttBridge(client_id="pc-face-tracker")
    bridge.connect()

    # Snap servo to center at start.
    bridge.publish_angle(cfg.servo_center_angle, force=True)

    cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Camera not available.")

    state = TrackState()

    show_debug = True
    frame_idx = 0
    last_status_t = 0.0

    # FPS
    t0 = time.time()
    fps_frames = 0
    fps: Optional[float] = None

    # Latest recognition results, keyed by face index. Reused between
    # recognition cycles so bounding boxes stay colored consistently.
    latest_identity: Dict[int, Tuple[Optional[str], float, bool]] = {}

    print("[track] running. q=quit | c=recenter | r=reload DB | d=debug")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if cfg.mirror_camera:
                frame = cv2.flip(frame, 1)
            h_full, w_full = frame.shape[:2]

            vis = frame.copy()
            cx_frame = w_full / 2.0

            # Dead-zone lines
            dz = cfg.tracking_dead_zone
            cv2.line(vis, (int(cx_frame), 0), (int(cx_frame), h_full), (255, 255, 0), 1)
            cv2.line(vis, (int(cx_frame - dz), 0), (int(cx_frame - dz), h_full), (128, 128, 0), 1)
            cv2.line(vis, (int(cx_frame + dz), 0), (int(cx_frame + dz), h_full), (128, 128, 0), 1)

            # -------- detection every frame --------
            faces = rec.detect(frame, max_faces=5)
            n_faces = len(faces)

            # -------- recognition every N frames --------
            run_recognition = (frame_idx % cfg.recognition_every_n_frames == 0)
            if run_recognition:
                latest_identity.clear()
                for i, f in enumerate(faces):
                    name, dist, accepted = rec.identify(frame, f)
                    latest_identity[i] = (name, dist, accepted)

            # -------- pick target --------
            target_face: Optional[FaceDet] = None
            target_index: Optional[int] = None
            target_name_for_frame: Optional[str] = None

            if state.target_name is not None:
                # Try to keep tracking the same identity.
                for i, f in enumerate(faces):
                    name, _, accepted = latest_identity.get(i, (None, 1.0, False))
                    if accepted and name == state.target_name:
                        target_face = f
                        target_index = i
                        target_name_for_frame = name
                        break

            # If no confirmed target yet, but a new identity is being voted on:
            if target_face is None:
                # Look for a vote winner in this recognition cycle
                vote_winner: Optional[str] = None
                vote_face: Optional[FaceDet] = None
                vote_index: Optional[int] = None
                for i, f in enumerate(faces):
                    name, _, accepted = latest_identity.get(i, (None, 1.0, False))
                    if accepted and name:
                        vote_winner = name
                        vote_face = f
                        vote_index = i
                        break

                if vote_winner is not None:
                    if state.pending_name == vote_winner:
                        state.confirmation_count += 1
                    else:
                        state.pending_name = vote_winner
                        state.confirmation_count = 1

                    if state.confirmation_count >= cfg.target_confirmation_frames:
                        # Confirm target
                        state.target_name = vote_winner
                        state.lost_counter = 0
                        target_face = vote_face
                        target_index = vote_index
                        target_name_for_frame = vote_winner
                        print(f"[track] CONFIRMED target: {vote_winner}")
                else:
                    state.pending_name = None
                    state.confirmation_count = 0

            # -------- update lost counter --------
            if target_face is not None:
                state.lost_counter = 0
            elif state.target_name is not None:
                state.lost_counter += 1
                if state.lost_counter > cfg.target_lost_frames:
                    print(f"[track] lost target: {state.target_name}")
                    state.reset()

            # -------- compute servo command --------
            if target_face is not None:
                cx = face_center_x(target_face)
                offset_px = cx - cx_frame
                state.last_offset_px = offset_px

                if abs(offset_px) < dz:
                    # inside dead-zone: hold
                    angle_target = state.last_angle
                else:
                    angle_target = angle_from_offset(offset_px, w_full)
            elif state.target_name is not None:
                # Target temporarily missing but not yet declared lost:
                # reuse last offset, but decay it toward center.
                state.last_offset_px *= 0.85
                angle_target = angle_from_offset(state.last_offset_px, w_full)
            else:
                # No target at all: drift toward center.
                state.last_offset_px *= 0.85
                if abs(state.last_offset_px) < 1.0:
                    state.last_offset_px = 0.0
                angle_target = angle_from_offset(state.last_offset_px, w_full)

            # Slew-limit the movement
            new_angle = slew_limit(state.last_angle, angle_target, cfg.tracking_max_step)
            state.last_angle = new_angle

            # Publish (MqttBridge enforces SERVO_MIN_CHANGE)
            bridge.publish_angle(new_angle)

            # -------- draw --------
            for i, f in enumerate(faces):
                name, dist, accepted = latest_identity.get(i, (None, 1.0, False))
                is_target = (target_index is not None and i == target_index)
                color = (0, 255, 0) if accepted else (0, 0, 255)
                if is_target:
                    color = (0, 255, 255)  # highlighted
                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, 2)
                for (x, y) in f.kps.astype(int):
                    cv2.circle(vis, (int(x), int(y)), 2, color, -1)

                if run_recognition:
                    label = name if accepted else "Unknown"
                    cv2.putText(vis, f"{label} d={dist:.2f}",
                                (f.x1, max(0, f.y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # HUD
            if show_debug:
                fps_frames += 1
                dt = time.time() - t0
                if dt >= 1.0:
                    fps = fps_frames / dt
                    fps_frames = 0
                    t0 = time.time()

                hud_lines = [
                    f"faces={n_faces}  target={state.target_name or '-'}  "
                    f"pending={state.pending_name or '-'}({state.confirmation_count})",
                    f"lost={state.lost_counter}/{cfg.target_lost_frames}  "
                    f"offset={state.last_offset_px:+.1f}px  angle={state.last_angle}",
                    f"fps={fps:.1f}" if fps else "fps=...",
                ]
                y = 28
                for line in hud_lines:
                    cv2.putText(vis, line, (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    y += 22

            cv2.imshow("track_with_recognition", vis)

            # -------- status/data publish (throttled) --------
            now = time.time()
            if now - last_status_t > 0.5:
                bridge.publish_status({
                    "has_target": state.target_name is not None,
                    "target": state.target_name,
                    "pending": state.pending_name,
                    "confirmation": state.confirmation_count,
                    "lost_frames": state.lost_counter,
                    "angle": state.last_angle,
                })
                bridge.publish_data({
                    "n_faces": n_faces,
                    "offset_px": round(state.last_offset_px, 1),
                    "angle": state.last_angle,
                    "recognized": [
                        {"name": n, "dist": round(d, 3)}
                        for (n, d, a) in latest_identity.values() if a
                    ],
                })
                last_status_t = now

            # -------- keys --------
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                state.last_offset_px = 0.0
                state.last_angle = cfg.servo_center_angle
                bridge.publish_angle(cfg.servo_center_angle, force=True)
                print("[track] recentered")
            elif key == ord("r"):
                rec.reload()
            elif key == ord("d"):
                show_debug = not show_debug
                print(f"[track] debug={'ON' if show_debug else 'OFF'}")

            frame_idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        bridge.disconnect()


if __name__ == "__main__":
    main()