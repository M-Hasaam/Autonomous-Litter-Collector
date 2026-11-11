"""View the ESP32-CAM over the USB cable (wired mode, no WiFi).

Firmware: USE_SERIAL_STREAM 1 in CameraWebServer.ino.
Frame format: A5 5A C3 3C | uint32 length (little-endian) | JPEG bytes

Usage:  py camera_serial.py [COM11] [baud]          wired, USB cable
        py camera_serial.py tcp:192.168.4.1:8080    wireless, through the ESP32 WiFi bridge
Add  --rotate ccw|cw|180  if the camera is mounted sideways or upside down (default: no rotation).
Keys in the window:  q = quit,  s = save the current frame as a JPEG into Embedded/esp32-cam/captures/
"""
import socket
import sys
import struct
import time
from pathlib import Path

import cv2
import numpy as np
import serial

import connection_help

# Optional "--rotate none|ccw|cw|180" (default none: the camera is mounted upright).
_ROTATIONS = {"none": None, "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE, "cw": cv2.ROTATE_90_CLOCKWISE, "180": cv2.ROTATE_180}
_argv = sys.argv[1:]
ROTATE = None
if "--rotate" in _argv:
    _i = _argv.index("--rotate")
    ROTATE = _ROTATIONS[_argv[_i + 1]]
    del _argv[_i:_i + 2]

PORT = _argv[0] if len(_argv) > 0 else "COM11"
BAUD = int(_argv[1]) if len(_argv) > 1 else 921_600
MAGIC = b"\xa5\x5a\xc3\x3c"
MAX_FRAME = 300_000
CAPTURE_DIR = Path(__file__).resolve().parents[3] / "esp32-cam" / "captures"
WINDOW = "ESP32-CAM"


def window_closed(name):
    """True once the window has been closed with the X button. Only call it after the first imshow()."""
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


class TcpSource:
    """Same read()/in_waiting/close() surface as a serial port, over a TCP socket."""

    def __init__(self, host, port):
        self.sock = socket.create_connection((host, port), timeout=5)
        self.sock.settimeout(0.05)
        self.in_waiting = 0

    def read(self, n):
        try:
            return self.sock.recv(65536)
        except socket.timeout:
            return b""

    def close(self):
        self.sock.close()


def open_port():
    if PORT.startswith("tcp:"):
        _, host, port = PORT.split(":")
        return TcpSource(host, int(port))
    ser = serial.Serial()
    ser.port = PORT
    ser.baudrate = BAUD
    ser.timeout = 0.05
    ser.dtr = False  # opening the port must not reset the board
    ser.rts = False
    ser.open()
    try:
        ser.set_buffer_size(rx_size=1 << 20)
    except Exception:
        pass
    return ser


def main():
    print("Connecting to the camera ...")
    try:
        ser = open_port()
    except OSError as exc:  # network errors and pyserial's SerialException are both OSError
        if PORT.startswith("tcp:"):
            _, host, port = PORT.split(":")
            print("\n" + connection_help.tcp_connect_message(exc, host, int(port)), file=sys.stderr)
        else:
            print("\n" + connection_help.serial_open_message(exc, PORT), file=sys.stderr)
        sys.exit(1)
    print(f"Connected to {PORT}. If no picture appears, press RESET on the board once.")
    buf = bytearray()
    frames = 0
    bad = 0
    last_report = time.perf_counter()
    last_count = 0
    fps = 0.0
    frame = None
    frame_time = 0.0
    last_jpeg = None
    saved = 0
    window_opened = False
    start_time = time.perf_counter()
    warned_no_frames = False

    try:
        while True:
            data = ser.read(max(ser.in_waiting, 1))
            if data:
                buf.extend(data)
                if len(buf) > 2_000_000:  # we are far behind: drop old data
                    buf = buf[-200_000:]

            # Parse as many complete frames as are buffered; keep only the newest to show
            while True:
                i = buf.find(MAGIC)
                if i == -1:
                    if len(buf) > 3:
                        del buf[:-3]
                    break
                if len(buf) < i + 8:
                    del buf[:i]
                    break
                (length,) = struct.unpack_from("<I", buf, i + 4)
                if length == 0 or length > MAX_FRAME:
                    del buf[:i + 4]
                    bad += 1
                    continue
                if len(buf) < i + 8 + length:
                    del buf[:i]
                    break
                jpeg = bytes(buf[i + 8:i + 8 + length])
                del buf[:i + 8 + length]
                if jpeg[:2] != b"\xff\xd8" or jpeg[-2:] != b"\xff\xd9":
                    bad += 1
                    continue
                img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    bad += 1
                    continue
                if ROTATE is not None:
                    img = cv2.rotate(img, ROTATE)
                frame = img
                last_jpeg = jpeg
                frame_time = time.perf_counter()
                frames += 1

            now = time.perf_counter()
            if frames == 0 and not warned_no_frames and now - start_time > 8:
                print("\n" + connection_help.NO_FRAMES_MESSAGE)
                warned_no_frames = True
            if now - last_report >= 1:
                fps = (frames - last_count) / (now - last_report)
                print(f"FPS: {fps:.1f} | Frames: {frames} | Bad frames: {bad}")
                last_count = frames
                last_report = now

            if frame is not None:
                shown = frame.copy()
                age_ms = (now - frame_time) * 1000
                cv2.putText(shown, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(shown, f"Read age: {age_ms:.0f} ms", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(WINDOW, shown)
                window_opened = True
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or (window_opened and window_closed(WINDOW)):  # q key or the X button
                break
            if key == ord("s") and last_jpeg is not None:
                # Save the camera's exact JPEG (unrotated) - ideal for collecting YOLO training images
                CAPTURE_DIR.mkdir(exist_ok=True)
                path = CAPTURE_DIR / f"frame_{time.strftime('%Y%m%d_%H%M%S')}_{saved:03d}.jpg"
                path.write_bytes(last_jpeg)
                saved += 1
                print(f"Saved {path}")
    except KeyboardInterrupt:
        print("\nStopped using Ctrl+C")
    except OSError as exc:  # connection reset, cable unplugged, ...
        print("\n" + connection_help.connection_lost_message(exc, PORT.startswith("tcp:")), file=sys.stderr)
    finally:
        ser.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
