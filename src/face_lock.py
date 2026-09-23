# src/face_lock.py
"""
Face Locking -- manual identity selection, lock persistence, and mandatory
action detection with a logged action history.

Built for the "Week-04 Face Locking" assignment. Reuses the existing
recognition/tracking/expression modules -- this file adds the three pieces
the assignment specifically grades:

  1. MANUAL SELECTION: at startup, you choose which enrolled identity to
     lock onto (not just "whoever the system happens to recognize first").
  2. ACTION DETECTION WHILE LOCKED: once locked, the system watches for
     mandatory actions -- face movement direction, blinks, smiles, and
     frowns -- and only for the locked identity.
  3. ACTION HISTORY LOG: every detected action is appended to a JSON
     Lines file with a timestamp, action type, and description
     (data/action_log/<name>.jsonl).

Lock behavior (matches the assignment's expected semantics):
  - Lock activates once the manually-selected identity is confirmed for
    TARGET_CONFIRMATION_FRAMES consecutive recognition cycles.
  - Once locked, other detected faces are ignored entirely.
  - The lock survives brief recognition failures (the target briefly not
    being confidently re-identified) and is only released after
    TARGET_LOST_FRAMES consecutive frames with no sign of the target.
  - CPU-only: onnxruntime defaults to CPUExecutionProvider (see embed.py),
    nothing here requires a GPU.

Run:
    python -m src.face_lock
Keys:
    q : quit
    c : recenter servo
    d : toggle debug overlay
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2

from .config import cfg
from .detect_landmarks import FaceDet
from .recognize import Recognizer
from .mqtt_bridge import MqttBridge
from .expressions import ExpressionAnalyzer, ExpressionResult


ACTION_LOG_DIR = Path("data/action_log")


# ============================================================
# ACTION LOGGING
# ============================================================

class ActionLogger:
    """Appends {timestamp, action, description} entries as JSON Lines."""

    def __init__(self, identity_name: str):
        ACTION_LOG_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(c if c.isalnum() else "_" for c in identity_name)
        self.path = ACTION_LOG_DIR / f"{safe_name}.jsonl"

    def log(self, action: str, description: str):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "action": action,
            "description": description,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[action] {entry['timestamp']}  {action:<12}  {description}")


# ============================================================
# LOCK STATE
# ============================================================

@dataclass
class LockState:
    locked_name: str                        # the manually-selected identity
    is_locked: bool = False
    confirmation_count: int = 0
    lost_counter: int = 0
    last_offset_px: float = 0.0
    last_angle: int = cfg.servo_center_angle
    last_direction: str = "center"           # left / right / center


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def face_center_x(f: FaceDet) -> float:
    return (f.x1 + f.x2) / 2.0


def angle_from_offset(offset_px: float, frame_w: float) -> int:
    half = frame_w / 2.0
    norm = clamp(offset_px / half, -1.0, 1.0)
    center = cfg.servo_center_angle
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


def direction_from_offset(offset_px: float, dead_zone: float) -> str:
    if offset_px > dead_zone:
        return "right"
    if offset_px < -dead_zone:
        return "left"
    return "center"


def expr_label(res: ExpressionResult) -> str:
    if not res.landmarks_found:
        return "-"
    if res.smile:
        return "Smiling"
    if res.frown:
        return "Frowning"
    if res.eyes_closed:
        return "Blinking"
    return "Neutral"


# ============================================================
# MANUAL SELECTION
# ============================================================

def select_identity(available_names) -> str:
    if not available_names:
        raise RuntimeError(
            "No enrolled identities found. Run `python -m src.enroll` first."
        )

    print("\nEnrolled identities:")
    for i, name in enumerate(available_names, start=1):
        print(f"  {i}. {name}")

    while True:
        choice = input("\nSelect the identity to lock onto (number or name): ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(available_names):
                return available_names[idx]
        elif choice in available_names:
            return choice
        print("Invalid selection, try again.")


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    print("[face_lock] loading recognizer...")
    rec = Recognizer()

    target_name = select_identity(rec.names)
    print(f"[face_lock] target identity: {target_name}")

    analyzer = ExpressionAnalyzer()
    logger = ActionLogger(target_name)
    bridge = MqttBridge(client_id="pc-face-lock")
    bridge.connect()

    bridge.publish_angle(cfg.servo_center_angle, force=True)

    cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Camera not available.")

    state = LockState(locked_name=target_name)

    show_debug = True
    frame_idx = 0
    last_status_t = 0.0

    t0 = time.time()
    fps_frames = 0
    fps: Optional[float] = None

    latest_identity: Dict[int, Tuple[Optional[str], float, bool]] = {}
    last_expr = ExpressionResult()
    prev_smile = False
    prev_frown = False

    logger.log("session_start", f"Locking session started for '{target_name}'.")
    print("[face_lock] running. q=quit | c=recenter | d=debug")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[face_lock] frame read failed, retrying...")
                time.sleep(0.1)
                continue

            if cfg.mirror_camera:
                frame = cv2.flip(frame, 1)
            h_full, w_full = frame.shape[:2]

            vis = frame.copy()
            cx_frame = w_full / 2.0
            dz = cfg.tracking_dead_zone
            cv2.line(vis, (int(cx_frame), 0), (int(cx_frame), h_full), (255, 255, 0), 1)

            # -------- detection every frame --------
            faces = rec.detect(frame, max_faces=5)

            # -------- recognition every N frames --------
            run_recognition = (frame_idx % cfg.recognition_every_n_frames == 0)
            if run_recognition:
                latest_identity.clear()
                for i, f in enumerate(faces):
                    name, dist, accepted = rec.identify(frame, f)
                    latest_identity[i] = (name, dist, accepted)

            # -------- find ONLY the manually-selected identity --------
            target_face: Optional[FaceDet] = None
            for i, f in enumerate(faces):
                name, _, accepted = latest_identity.get(i, (None, 1.0, False))
                if accepted and name == target_name:
                    target_face = f
                    break

            # -------- lock activation (manual target, confirmation-gated) --------
            if not state.is_locked:
                if target_face is not None:
                    state.confirmation_count += 1
                    if state.confirmation_count >= cfg.target_confirmation_frames:
                        state.is_locked = True
                        state.lost_counter = 0
                        logger.log("lock_acquired", f"Locked onto '{target_name}'.")
                else:
                    state.confirmation_count = 0

            # -------- lock persistence / release --------
            if state.is_locked:
                if target_face is not None:
                    state.lost_counter = 0
                else:
                    state.lost_counter += 1
                    if state.lost_counter > cfg.target_lost_frames:
                        logger.log("lock_released",
                                    f"Lost '{target_name}' for {state.lost_counter} frames.")
                        state.is_locked = False
                        state.confirmation_count = 0
                        state.lost_counter = 0

            # -------- servo + movement-direction action (only while locked) --------
            if state.is_locked and target_face is not None:
                cx = face_center_x(target_face)
                offset_px = cx - cx_frame
                state.last_offset_px = offset_px

                new_direction = direction_from_offset(offset_px, dz)
                if new_direction != state.last_direction:
                    logger.log("movement",
                                f"'{target_name}' moved {new_direction} of center.")
                    state.last_direction = new_direction

                angle_target = state.last_angle if new_direction == "center" \
                    else angle_from_offset(offset_px, w_full)
            elif state.is_locked:
                state.last_offset_px *= 0.85
                angle_target = angle_from_offset(state.last_offset_px, w_full)
            else:
                state.last_offset_px *= 0.85
                if abs(state.last_offset_px) < 1.0:
                    state.last_offset_px = 0.0
                angle_target = angle_from_offset(state.last_offset_px, w_full)

            new_angle = slew_limit(state.last_angle, angle_target, cfg.tracking_max_step)
            state.last_angle = new_angle
            bridge.publish_angle(new_angle)

            # -------- action detection (blink/smile/frown), only while locked --------
            if state.is_locked and target_face is not None:
                box = (target_face.x1, target_face.y1, target_face.x2, target_face.y2)
                last_expr = analyzer.analyze(frame, box)

                if last_expr.blink_event:
                    logger.log("blink", f"'{target_name}' blinked.")

                if last_expr.smile and not prev_smile:
                    logger.log("smile", f"'{target_name}' started smiling.")
                prev_smile = last_expr.smile

                if last_expr.frown and not prev_frown:
                    logger.log("frown", f"'{target_name}' started frowning.")
                prev_frown = last_expr.frown

            # -------- draw --------
            for i, f in enumerate(faces):
                name, dist, accepted = latest_identity.get(i, (None, 1.0, False))
                is_target = (target_face is not None and f is target_face)

                if state.is_locked and not is_target:
                    continue  # locked: ignore every other face visually too

                color = (0, 255, 255) if is_target else (0, 255, 0) if accepted else (0, 0, 255)
                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, 2)

                if run_recognition:
                    label = name if accepted else "Unknown"
                    cv2.putText(vis, f"{label} d={dist:.2f}",
                                (f.x1, max(0, f.y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if is_target:
                    cv2.putText(vis, expr_label(last_expr),
                                (f.x1, min(h_full - 5, f.y2 + 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # -------- HUD --------
            if show_debug:
                fps_frames += 1
                dt = time.time() - t0
                if dt >= 1.0:
                    fps = fps_frames / dt
                    fps_frames = 0
                    t0 = time.time()

                hud_lines = [
                    f"target={target_name}  locked={'YES' if state.is_locked else 'no'}  "
                    f"confirm={state.confirmation_count}/{cfg.target_confirmation_frames}",
                    f"lost={state.lost_counter}/{cfg.target_lost_frames}  "
                    f"dir={state.last_direction}  angle={state.last_angle}",
                    f"expr={expr_label(last_expr)}  blinks={analyzer.blink_count}",
                    f"fps={fps:.1f}" if fps else "fps=...",
                ]
                y = 28
                for line in hud_lines:
                    cv2.putText(vis, line, (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    y += 22

            cv2.imshow("face_lock", vis)

            # -------- MQTT status publish (throttled) --------
            now = time.time()
            if now - last_status_t > 0.5:
                bridge.publish_status({
                    "target": target_name,
                    "locked": state.is_locked,
                    "direction": state.last_direction,
                    "angle": state.last_angle,
                    "expression": expr_label(last_expr),
                    "blink_count": analyzer.blink_count,
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
                print("[face_lock] recentered")
            elif key == ord("d"):
                show_debug = not show_debug

            frame_idx += 1

    finally:
        logger.log("session_end", f"Locking session ended for '{target_name}'.")
        cap.release()
        cv2.destroyAllWindows()
        analyzer.close()
        bridge.disconnect()


if __name__ == "__main__":
    main()