# src/tracker.py

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrackingResult:
    face_x: float
    frame_center_x: float
    error_x: float
    normalized_error: float
    servo_angle: int
    moving: bool


class HorizontalTracker:
    """
    Horizontal face tracker.

    The camera image is treated as the control surface.

    If the target is:
        LEFT  -> servo moves left
        RIGHT -> servo moves right
        CENTER -> servo stops

    The controller uses proportional control:

        movement = Kp * error

    A dead zone prevents servo jitter.
    """

    def __init__(
        self,
        center_angle: int = 90,
        min_angle: int = 10,
        max_angle: int = 170,
        dead_zone: int = 35,
        kp: float = 0.08,
        max_step: int = 4,
        min_change: int = 1,
    ):
        self.center_angle = int(center_angle)
        self.min_angle = int(min_angle)
        self.max_angle = int(max_angle)

        self.dead_zone = float(dead_zone)
        self.kp = float(kp)
        self.max_step = int(max_step)
        self.min_change = int(min_change)

        self.current_angle = self.center_angle

    def reset(self, angle: int | None = None) -> int:
        if angle is None:
            angle = self.center_angle

        self.current_angle = self._clamp_angle(angle)
        return self.current_angle

    def _clamp_angle(self, angle: float) -> int:
        return int(
            max(
                self.min_angle,
                min(self.max_angle, round(angle)),
            )
        )

    def update(
        self,
        face_x: float,
        frame_width: int,
    ) -> TrackingResult:

        frame_center_x = frame_width / 2.0

        error_x = float(face_x - frame_center_x)

        normalized_error = error_x / max(frame_center_x, 1.0)

        # Target is sufficiently centered.
        if abs(error_x) <= self.dead_zone:
            return TrackingResult(
                face_x=face_x,
                frame_center_x=frame_center_x,
                error_x=error_x,
                normalized_error=normalized_error,
                servo_angle=self.current_angle,
                moving=False,
            )

        # Proportional movement.
        step = self.kp * error_x

        # Limit how quickly the servo can move.
        step = max(
            -self.max_step,
            min(self.max_step, step),
        )

        # Camera/servo direction:
        #
        # Positive image error = face is RIGHT.
        # Positive servo angle = rotate RIGHT.
        #
        # If your physical servo behaves in the opposite
        # direction, reverse this sign.
        new_angle = self.current_angle + step

        new_angle = self._clamp_angle(new_angle)

        moving = abs(new_angle - self.current_angle) >= self.min_change

        if moving:
            self.current_angle = new_angle

        return TrackingResult(
            face_x=face_x,
            frame_center_x=frame_center_x,
            error_x=error_x,
            normalized_error=normalized_error,
            servo_angle=self.current_angle,
            moving=moving,
        )