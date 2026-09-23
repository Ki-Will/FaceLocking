# src/mqtt_bridge.py
"""
MQTT bridge for face-tracking servo control.
Reads configuration from src/config.py (which loads .env).
"""
from __future__ import annotations

import json
import time
from typing import Optional

import paho.mqtt.client as mqtt

from .config import cfg


class MqttBridge:
    def __init__(self, client_id: str = "pc-face-tracker"):
        self.client = mqtt.Client(client_id=client_id, clean_session=True)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

        self._connected = False
        self._last_angle: Optional[int] = None
        self._last_publish_time = 0.0

    # ---------------- callbacks ----------------
    def _on_connect(self, client, userdata, flags, rc):
        self._connected = (rc == 0)
        print(f"[MQTT] connect rc={rc} to {cfg.mqtt_broker}:{cfg.mqtt_port}")

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        print(f"[MQTT] disconnected rc={rc}")

    # ---------------- lifecycle ----------------
    def connect(self):
        self.client.connect(cfg.mqtt_broker, cfg.mqtt_port, cfg.mqtt_keepalive)
        self.client.loop_start()
        t0 = time.time()
        while not self._connected and time.time() - t0 < 5.0:
            time.sleep(0.05)
        if not self._connected:
            raise RuntimeError("MQTT connection timed out")

    def disconnect(self):
        try:
            self.client.loop_stop()
        except Exception:
            pass
        try:
            self.client.disconnect()
        except Exception:
            pass

    # ---------------- publishing ----------------
    def publish_angle(
        self,
        angle: int,
        force: bool = False,
        min_interval_s: float = 0.05,
    ) -> bool:
        """
        Publish a servo angle (integer string).
        Returns True if published, False if suppressed.

        Suppression rules (unless force=True):
          - identical to last published angle
          - differs from last by less than cfg.servo_min_change
          - within min_interval_s of the last publish
        """
        angle = int(angle)

        # Hard clamp to configured range
        angle = max(cfg.servo_min_angle, min(cfg.servo_max_angle, angle))

        if not force:
            if self._last_angle is not None:
                if abs(angle - self._last_angle) < cfg.servo_min_change:
                    return False
            if (time.time() - self._last_publish_time) < min_interval_s:
                return False

        self.client.publish(cfg.mqtt_servo_topic, str(angle), qos=0, retain=False)
        self._last_angle = angle
        self._last_publish_time = time.time()
        print(f"[MQTT] -> {cfg.mqtt_servo_topic} = {angle}")
        return True

    def publish_status(self, payload: dict):
        self.client.publish(
            cfg.mqtt_status_topic, json.dumps(payload), qos=0, retain=False
        )

    def publish_data(self, payload: dict):
        self.client.publish(
            cfg.mqtt_data_topic, json.dumps(payload), qos=0, retain=False
        )