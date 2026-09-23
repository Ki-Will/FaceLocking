# esp_logger.py
"""
Live serial logger for the ESP8266 servo controller.
Run: python esp_logger.py
Press Ctrl+C to stop.
"""

import serial
import time

# -------- CONFIG --------
PORT = "COM5"        # Change to your ESP8266 port
BAUD = 115200        # MicroPython REPL baud rate

def main():
    print(f"Opening {PORT} at {BAUD} baud...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
    except serial.SerialException as e:
        print(f"Failed to open port: {e}")
        return

    print("Listening for ESP8266 logs. Press Ctrl+C to exit.")
    print("-" * 50)

    try:
        while True:
            # readline() blocks up to 1s and returns b'' on timeout
            line = ser.readline()
            if line:
                try:
                    text = line.decode('utf-8', errors='replace').rstrip()
                    print(text)
                except Exception:
                    # Ignore decode errors (e.g., boot noise)
                    pass
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        ser.close()
        print("Serial port closed.")

if __name__ == "__main__":
    main()