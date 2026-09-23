# esp8266/umqtt_simple.py

import socket
import struct
import time


class MQTTException(Exception):
    pass


class MQTTClient:

    def __init__(
        self,
        client_id,
        server,
        port=1883,
        user=None,
        password=None,
        keepalive=60,
    ):
        self.client_id = client_id
        self.server = server
        self.port = port
        self.user = user
        self.password = password
        self.keepalive = keepalive

        self.sock = None
        self.cb = None

    def _encode_str(self, value):
        encoded = value.encode()
        return struct.pack("!H", len(encoded)) + encoded

    def _encode_remaining_length(
        self,
        length,
    ):
        output = bytearray()

        while True:

            digit = length % 128
            length //= 128

            if length > 0:
                digit |= 0x80

            output.append(digit)

            if length == 0:
                break

        return output

    def _wait_msg(self):
        first_byte = self.sock.read(1)

        if not first_byte:
            return None

        byte1 = first_byte[0]

        msg_type = byte1 >> 4
        msg_qos = (
            byte1 >> 1
        ) & 0x03

        remaining_length = 0
        multiplier = 1

        while True:

            digit = self.sock.read(1)

            if not digit:
                raise MQTTException(
                    "Connection closed"
                )

            digit = digit[0]

            remaining_length += (
                (digit & 127)
                * multiplier
            )

            if not (
                digit & 128
            ):
                break

            multiplier *= 128

            if multiplier > 128 * 128 * 128:
                raise MQTTException(
                    "Malformed remaining length"
                )

        payload = b""

        if remaining_length:
            payload = self.sock.read(
                remaining_length
            )

        return (
            msg_type,
            msg_qos,
            payload,
        )

    def connect(self):
        addr = socket.getaddrinfo(
            self.server,
            self.port,
        )[0][-1]

        self.sock = socket.socket()
        self.sock.settimeout(5)
        self.sock.connect(addr)

        flags = 0x02

        if self.user is not None:
            flags |= 0x80
            if self.password is not None:
                flags |= 0x40

        var_header = bytearray()
        var_header.extend(b"\x00\x04MQTT")
        var_header.append(0x04)
        var_header.append(flags)
        var_header.extend(struct.pack("!H", self.keepalive))

        payload = bytearray()
        payload.extend(self._encode_str(self.client_id))

        if self.user is not None:
            payload.extend(self._encode_str(self.user))
            if self.password is not None:
                payload.extend(self._encode_str(self.password))

        packet = var_header + payload
        remaining = len(packet)

        self.sock.write(b"\x10")
        self.sock.write(self._encode_remaining_length(remaining))
        self.sock.write(packet)

        response = self._wait_msg()

        if response is None:
            raise MQTTException("No MQTT CONNACK")

        msg_type, _, payload = response

        if msg_type != 2 or len(payload) < 2:
            raise MQTTException("Invalid CONNACK")

        return payload[1]
    def set_callback(
        self,
        callback,
    ):
        self.cb = callback

    def publish(
        self,
        topic,
        msg,
        qos=0,
        retain=False,
    ):

        if isinstance(
            msg,
            str,
        ):
            msg = msg.encode()

        topic_bytes = topic.encode()

        packet = bytearray()

        packet.extend(
            struct.pack(
                "!H",
                len(topic_bytes),
            )
        )

        packet.extend(
            topic_bytes
        )

        packet.extend(
            msg
        )

        header = 0x30

        if retain:
            header |= 0x01

        self.sock.write(
            bytes([header])
        )

        self.sock.write(
            self._encode_remaining_length(
                len(packet)
            )
        )

        self.sock.write(
            packet
        )

    def subscribe(
        self,
        topic,
        qos=0,
    ):

        packet_id = 1

        topic_bytes = topic.encode()

        payload = bytearray()

        payload.extend(
            struct.pack(
                "!H",
                packet_id,
            )
        )

        payload.extend(
            struct.pack(
                "!H",
                len(topic_bytes),
            )
        )

        payload.extend(
            topic_bytes
        )

        payload.append(
            qos
        )

        self.sock.write(
            b"\x82"
        )

        self.sock.write(
            self._encode_remaining_length(
                len(payload)
            )
        )

        self.sock.write(
            payload
        )

    def check_msg(self):

        if self.sock is None:
            return

        self.sock.settimeout(
            0.01
        )

        try:

            msg = self._wait_msg()

            if msg is None:
                return

            msg_type, _, payload = msg

            # PUBLISH
            if msg_type == 3:

                topic_len = (
                    payload[0] << 8
                    | payload[1]
                )

                topic_start = 2

                topic_end = (
                    topic_start
                    + topic_len
                )

                topic = payload[
                    topic_start:topic_end
                ].decode()

                message = payload[
                    topic_end:
                ]

                if self.cb:
                    self.cb(
                        topic,
                        message,
                    )

            # PINGRESP
            elif msg_type == 13:
                pass

        except OSError:
            pass

        finally:

            self.sock.settimeout(
                5
            )

    def ping(self):

        self.sock.write(
            b"\xc0\x00"
        )

    def disconnect(self):

        if self.sock:

            try:
                self.sock.write(
                    b"\xe0\x00"
                )
            except Exception:
                pass

            try:
                self.sock.close()
            except Exception:
                pass

            self.sock = None