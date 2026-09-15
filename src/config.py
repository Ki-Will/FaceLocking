# src/config.py
"""
Central configuration loaded from .env.
Import `cfg` anywhere you need settings.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load once at import time.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _get_str(key: str, default: str) -> str:
    v = os.getenv(key)
    return default if v is None or v == "" else v


def _get_int(key: str, default: int) -> int:
    try:
        return int(_get_str(key, str(default)))
    except ValueError:
        return default


def _get_float(key: str, default: float) -> float:
    try:
        return float(_get_str(key, str(default)))
    except ValueError:
        return default


def _get_bool(key: str, default: bool) -> bool:
    v = _get_str(key, "true" if default else "false").strip().lower()
    return v in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Config:
    # Camera
    camera_index: int
    mirror_camera: bool

    # Model
    arc_face_model: str
    recognition_threshold: float

    # MQTT
    mqtt_broker: str
    mqtt_port: int
    mqtt_keepalive: int
    mqtt_servo_topic: str
    mqtt_status_topic: str
    mqtt_data_topic: str

    # Servo
    servo_min_angle: int
    servo_max_angle: int
    servo_center_angle: int
    servo_min_change: int

    # Tracking
    tracking_dead_zone: int
    tracking_kp: float
    tracking_max_step: int

    # Recognition performance
    recognition_every_n_frames: int
    target_confirmation_frames: int
    target_lost_frames: int


cfg = Config(
    camera_index=_get_int("CAMERA_INDEX", 0),
    mirror_camera=_get_bool("MIRROR_CAMERA", True),

    arc_face_model=_get_str("ARC_FACE_MODEL", "models/embedder_arcface.onnx"),
    recognition_threshold=_get_float("RECOGNITION_THRESHOLD", 0.34),

    mqtt_broker=_get_str("MQTT_BROKER", "broker.hivemq.com"),
    mqtt_port=_get_int("MQTT_PORT", 1883),
    mqtt_keepalive=_get_int("MQTT_KEEPALIVE", 120),
    mqtt_servo_topic=_get_str("MQTT_SERVO_TOPIC", "face/servo/angle"),
    mqtt_status_topic=_get_str("MQTT_STATUS_TOPIC", "face/tracking/status"),
    mqtt_data_topic=_get_str("MQTT_DATA_TOPIC", "face/tracking/data"),

    servo_min_angle=_get_int("SERVO_MIN_ANGLE", 10),
    servo_max_angle=_get_int("SERVO_MAX_ANGLE", 170),
    servo_center_angle=_get_int("SERVO_CENTER_ANGLE", 90),
    servo_min_change=_get_int("SERVO_MIN_CHANGE", 1),

    tracking_dead_zone=_get_int("TRACKING_DEAD_ZONE", 35),
    tracking_kp=_get_float("TRACKING_KP", 0.08),
    tracking_max_step=_get_int("TRACKING_MAX_STEP", 4),

    recognition_every_n_frames=_get_int("RECOGNITION_EVERY_N_FRAMES", 5),
    target_confirmation_frames=_get_int("TARGET_CONFIRMATION_FRAMES", 3),
    target_lost_frames=_get_int("TARGET_LOST_FRAMES", 15),
)