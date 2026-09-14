from __future__ import annotations

import json
import time

import paho.mqtt.client as mqtt


class ServoController:

    def __init__(
        self,
        broker="broker.hivemq.com",
        port=1883,
        servo_topic="face/servo/angle",
        status_topic="face/tracking/status",
        data_topic="face/tracking/data",
    ):

        self.broker = broker
        self.port = port

        self.servo_topic = (
            servo_topic
        )

        self.status_topic = (
            status_topic
        )

        self.data_topic = (
            data_topic
        )

        self.connected = False
        self.last_angle = None

        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=(
                f"face-tracker-"
                f"{int(time.time())}"
            ),
        )

        self.client.on_connect = (
            self._on_connect
        )

        self.client.on_disconnect = (
            self._on_disconnect
        )

    def _on_connect(
        self,
        client,
        userdata,
        flags,
        reason_code,
        properties,
    ):

        if reason_code == 0:

            self.connected = True

            print(
                "[MQTT] Connected"
            )

        else:

            print(
                "[MQTT] Connection failed:",
                reason_code,
            )

    def _on_disconnect(
        self,
        client,
        userdata,
        disconnect_flags,
        reason_code,
        properties,
    ):

        self.connected = False

        print(
            "[MQTT] Disconnected:",
            reason_code,
        )

    def connect(self):

        self.client.connect(
            self.broker,
            self.port,
            60,
        )

        self.client.loop_start()

        deadline = (
            time.time() + 5
        )

        while (
            not self.connected
            and time.time()
            < deadline
        ):

            time.sleep(
                0.05
            )

        if not self.connected:

            raise RuntimeError(
                "MQTT connection failed."
            )

    def publish_angle(
        self,
        angle,
    ):

        if not self.connected:
            return

        angle = max(
            0,
            min(
                180,
                int(angle),
            ),
        )

        if (
            self.last_angle
            == angle
        ):
            return

        self.client.publish(
            self.servo_topic,
            str(angle),
        )

        self.last_angle = angle

    def publish_status(
        self,
        status,
    ):

        if not self.connected:
            return

        self.client.publish(
            self.status_topic,
            status,
        )

    def publish_tracking_data(
        self,
        name,
        face_x,
        frame_center_x,
        error_x,
        servo_angle,
        distance=None,
        similarity=None,
    ):

        if not self.connected:
            return

        payload = {
            "name": name,
            "face_x": face_x,
            "frame_center_x": frame_center_x,
            "error_x": error_x,
            "servo_angle": servo_angle,
            "distance": distance,
            "similarity": similarity,
            "timestamp": time.time(),
        }

        self.client.publish(
            self.data_topic,
            json.dumps(payload),
        )

    def disconnect(self):

        try:

            self.client.loop_stop()

            self.client.disconnect()

        except Exception:

            pass

        self.connected = False