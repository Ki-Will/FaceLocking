# esp8266/main.py

import time

from machine import Pin, PWM
import network

from umqtt_simple import MQTTClient


# ============================================================
# WIFI
# ============================================================

WIFI_SSID = "Ngaho_ra"
WIFI_PASSWORD = "icyayi_cya_mukaru"


# ============================================================
# MQTT
# ============================================================

MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883

MQTT_CLIENT_ID = (
    "esp8266-face-servo"
)

SERVO_TOPIC = (
    "face/servo/angle"
)

STATUS_TOPIC = (
    "face/tracking/status"
)


# ============================================================
# SERVO
# ============================================================

# D5 on NodeMCU / Wemos ESP8266
# = GPIO14

SERVO_PIN = 5

SERVO_MIN_ANGLE = 10
SERVO_MAX_ANGLE = 170
SERVO_START_ANGLE = 90


# Standard servo PWM
SERVO_FREQUENCY = 50


# ============================================================
# SERVO OBJECT
# ============================================================

servo = PWM(
    Pin(SERVO_PIN),
    freq=SERVO_FREQUENCY,
)


current_angle = (
    SERVO_START_ANGLE
)


def clamp(
    value,
    minimum,
    maximum,
):
    return max(
        minimum,
        min(
            maximum,
            value,
        ),
    )


def angle_to_duty(
    angle,
):

    angle = clamp(
        angle,
        SERVO_MIN_ANGLE,
        SERVO_MAX_ANGLE,
    )

    # Approximate servo pulse:
    #
    # 0 degrees   -> 500 us
    # 180 degrees -> 2500 us
    #
    # 20 ms period at 50 Hz.

    pulse_us = (
        500
        + (
            angle * 2000
            // 180
        )
    )

    duty = int(
        pulse_us
        * 1023
        // 20000
    )

    return duty


def set_angle(
    angle,
):

    global current_angle

    angle = int(
        clamp(
            angle,
            SERVO_MIN_ANGLE,
            SERVO_MAX_ANGLE,
        )
    )

    duty = angle_to_duty(
        angle
    )

    servo.duty(
        duty
    )

    current_angle = angle

    print(
        "[SERVO] angle:",
        angle,
        "duty:",
        duty,
    )


# ============================================================
# WIFI
# ============================================================

def connect_wifi():

    wlan = network.WLAN(
        network.STA_IF
    )

    wlan.active(
        True
    )

    if wlan.isconnected():

        print(
            "[WiFi] Already connected"
        )

        print(
            "[WiFi]",
            wlan.ifconfig(),
        )

        return wlan

    print(
        "[WiFi] Connecting..."
    )

    wlan.connect(
        WIFI_SSID,
        WIFI_PASSWORD,
    )

    timeout = 20

    while (
        not wlan.isconnected()
        and timeout > 0
    ):

        time.sleep(1)

        print(
            "[WiFi] waiting..."
        )

        timeout -= 1

    if not wlan.isconnected():

        raise RuntimeError(
            "WiFi connection failed"
        )

    print(
        "[WiFi] Connected"
    )

    print(
        "[WiFi]",
        wlan.ifconfig(),
    )

    return wlan


# ============================================================
# MQTT CALLBACK
# ============================================================

def mqtt_callback(
    topic,
    message,
):

    global current_angle

    try:

        topic_text = (
            topic.decode()
            if isinstance(
                topic,
                bytes,
            )
            else topic
        )

        message_text = (
            message.decode()
            if isinstance(
                message,
                bytes,
            )
            else message
        )

        print(
            "[MQTT]",
            topic_text,
            "->",
            message_text,
        )

        if topic_text == SERVO_TOPIC:

            try:

                angle = int(
                    float(
                        message_text
                    )
                )

                set_angle(
                    angle
                )

            except ValueError:

                print(
                    "[SERVO] Invalid angle:",
                    message_text,
                )

    except Exception as exc:

        print(
            "[MQTT] Callback error:",
            exc,
        )


# ============================================================
# MQTT CONNECT
# ============================================================

def connect_mqtt():

    print(
        "[MQTT] Connecting..."
    )

    client = MQTTClient(
        MQTT_CLIENT_ID,
        MQTT_BROKER,
        port=MQTT_PORT,
        keepalive=60,
    )

    client.set_callback(
        mqtt_callback
    )

    result = client.connect()

    print(
        "[MQTT] Connected, result:",
        result,
    )

    client.subscribe(
        SERVO_TOPIC
    )

    print(
        "[MQTT] Subscribed:",
        SERVO_TOPIC,
    )

    return client


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "================================"
    )
    print(
        " ESP8266 FACE SERVO CONTROLLER"
    )
    print(
        "================================"
    )
    print(
        "Servo pin: D5 / GPIO14"
    )
    print(
        "MQTT broker:",
        MQTT_BROKER,
    )
    print()

    set_angle(
        SERVO_START_ANGLE
    )

    connect_wifi()

    client = connect_mqtt()

    print()
    print(
        "[SYSTEM] Ready."
    )
    print()

    last_ping = time.time()

    while True:

        try:

            client.check_msg()

            # MQTT keepalive
            if (
                time.time()
                - last_ping
                > 30
            ):

                try:
                    client.ping()
                except Exception:
                    pass

                last_ping = (
                    time.time()
                )

            time.sleep(
                0.1
            )

        except Exception as exc:

            print(
                "[SYSTEM] Error:",
                exc,
            )

            print(
                "[SYSTEM] Reconnecting..."
            )

            time.sleep(
                3
            )

            try:

                connect_wifi()

                client = connect_mqtt()

                set_angle(
                    current_angle
                )

            except Exception as reconnect_error:

                print(
                    "[SYSTEM] Reconnect failed:",
                    reconnect_error,
                )

                time.sleep(
                    5
                )


try:

    main()

except KeyboardInterrupt:

    print(
        "[SYSTEM] Stopped"
    )

finally:

    try:
        servo.deinit()
    except Exception:
        pass