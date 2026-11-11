"""Run the app's YOLO model (litter_detector.tflite) on the ESP32-CAM stream, on the PC.

It does exactly what the Android app does (YoloDetector.kt): letterbox to 640x640 with gray 114 padding,
RGB scaled to 0..1, decode [1, 4+classes, anchors], confidence 0.25, per-class NMS at IoU 0.45.

Needs:  py -m pip install ai-edge-litert opencv-python numpy pyserial

Usage:
  py camera_yolo.py                              live window, stream from the ESP32 WiFi bridge
  py camera_yolo.py --source COM11 --baud 921600 stream over the USB cable instead
  py camera_yolo.py --headless --seconds 30      no window, print a summary (for automated tests)
Keys in the window:  q = quit,  s = save the current annotated frame into ESP_CAM/captures/
"""
import argparse
import socket
import struct
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

import connection_help

ROOT = Path(__file__).resolve().parents[2]
MAGIC = b"\xa5\x5a\xc3\x3c"
MAX_FRAME = 300_000
WINDOW = "ESP32-CAM + YOLO (app model)"


def window_closed(name):
    """True once the window has been closed with the X button. Only call it after the first imshow()."""
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


ROTATIONS = {
    "none": None,
    "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,  # same as camera_serial.py
    "cw": cv2.ROTATE_90_CLOCKWISE,
    "180": cv2.ROTATE_180,
}


# ---------------------------------------------------------------- frame source (same format as camera_serial.py)
class Source:
    def __init__(self, spec, baud):
        self.sock = None
        self.ser = None
        if spec.startswith("tcp:"):
            _, host, port = spec.split(":")
            self.sock = socket.create_connection((host, int(port)), timeout=5)
            self.sock.settimeout(0.05)
        else:
            import serial
            self.ser = serial.Serial()
            self.ser.port, self.ser.baudrate, self.ser.timeout = spec, baud, 0.05
            self.ser.dtr = False  # opening the port must not reset the board
            self.ser.rts = False
            self.ser.open()

    def read(self):
        if self.sock:
            try:
                return self.sock.recv(65536)
            except socket.timeout:
                return b""
        return self.ser.read(max(self.ser.in_waiting, 1))

    def close(self):
        (self.sock or self.ser).close()


class FrameGrabber(threading.Thread):
    """Reads the byte stream in the background and always keeps only the newest JPEG."""

    def __init__(self, source):
        super().__init__(daemon=True)
        self.source = source
        self.lock = threading.Lock()
        self.jpeg = None
        self.count = 0
        self.bad = 0
        self.running = True
        self.error = None

    def run(self):
        buf = bytearray()
        try:
            while self.running:
                data = self.source.read()
                if data:
                    buf.extend(data)
                    if len(buf) > 2_000_000:
                        buf = buf[-200_000:]
                while True:
                    i = buf.find(MAGIC)
                    if i == -1:
                        if len(buf) > 3:
                            del buf[:-3]
                        break
                    if len(buf) < i + 8:
                        del buf[:i]
                        break
                    (n,) = struct.unpack_from("<I", buf, i + 4)
                    if n == 0 or n > MAX_FRAME:
                        del buf[:i + 4]
                        self.bad += 1
                        continue
                    if len(buf) < i + 8 + n:
                        del buf[:i]
                        break
                    jpeg = bytes(buf[i + 8:i + 8 + n])
                    del buf[:i + 8 + n]
                    if jpeg[:2] != b"\xff\xd8" or jpeg[-2:] != b"\xff\xd9":
                        self.bad += 1
                        continue
                    with self.lock:
                        self.jpeg = jpeg
                        self.count += 1
        except Exception as exc:  # connection lost, port unplugged...
            self.error = exc
            self.running = False

    def latest(self):
        with self.lock:
            return self.jpeg, self.count


# ---------------------------------------------------------------- YOLO, same steps as YoloDetector.kt
class Yolo:
    def __init__(self, model_path, labels_path, conf, iou, threads):
        self.labels = [l.strip() for l in Path(labels_path).read_text().splitlines() if l.strip()]
        self.interp = Interpreter(model_path=str(model_path), num_threads=threads)
        self.interp.allocate_tensors()
        inp = self.interp.get_input_details()[0]
        out = self.interp.get_output_details()[0]
        self.in_index, self.out_index = inp["index"], out["index"]
        _, self.size_h, self.size_w, _ = [int(v) for v in inp["shape"]]
        shape = [int(v) for v in out["shape"]]
        self.channels_first = shape[1] == 4 + len(self.labels)
        self.conf, self.iou = conf, iou

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        scale = min(self.size_w / w, self.size_h / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        pad_x, pad_y = (self.size_w - nw) / 2.0, (self.size_h - nh) / 2.0
        canvas = np.full((self.size_h, self.size_w, 3), 114, np.uint8)
        x0, y0 = int(round(pad_x)), int(round(pad_y))
        canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        self.interp.set_tensor(self.in_index, x[None])
        self.interp.invoke()
        out = self.interp.get_tensor(self.out_index)[0]
        out = out if self.channels_first else out.T  # -> [4 + classes, anchors]

        scores = out[4:]
        cls = scores.argmax(axis=0)
        best = scores[cls, np.arange(scores.shape[1])]
        keep = best > self.conf
        if not keep.any():
            return []
        cx, cy, bw, bh = out[0][keep], out[1][keep], out[2][keep], out[3][keep]
        cls, best = cls[keep], best[keep]
        left = np.clip((cx - bw / 2 - pad_x) / scale, 0, w)
        top = np.clip((cy - bh / 2 - pad_y) / scale, 0, h)
        right = np.clip((cx + bw / 2 - pad_x) / scale, 0, w)
        bottom = np.clip((cy + bh / 2 - pad_y) / scale, 0, h)
        dets = []
        for c in np.unique(cls):  # per-class NMS, highest confidence first
            idx = np.where(cls == c)[0]
            boxes = [[float(left[i]), float(top[i]), float(right[i] - left[i]), float(bottom[i] - top[i])] for i in idx]
            for k in np.array(cv2.dnn.NMSBoxes(boxes, best[idx].tolist(), self.conf, self.iou)).flatten():
                i = idx[int(k)]
                if right[i] > left[i] and bottom[i] > top[i]:
                    dets.append((self.labels[int(c)], float(best[i]), int(left[i]), int(top[i]), int(right[i]), int(bottom[i])))
        return sorted(dets, key=lambda d: -d[1])


def draw(frame, dets, fps, infer_ms, grab_fps):
    for label, score, x1, y1, x2, y2 in dets:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} {int(score * 100)}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(frame, text, (x1 + 2, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    cv2.putText(frame, f"YOLO {fps:.1f}/s  infer {infer_ms:.0f} ms  camera {grab_fps:.1f} FPS  objects {len(dets)}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="tcp:192.168.4.1:8080", help="tcp:HOST:PORT (bridge) or a COM port")
    ap.add_argument("--baud", type=int, default=921_600, help="only for a COM port source")
    ap.add_argument("--model", default=str(ROOT / "app/src/main/assets/litter_detector.tflite"))
    ap.add_argument("--labels", default=str(ROOT / "app/src/main/assets/labels.txt"))
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold (the app uses 0.25)")
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--rotate", choices=ROTATIONS, default="none",
                    help="turn the image upright before detection (default none: camera mounted upright; "
                         "use ccw or cw if it is mounted sideways)")
    ap.add_argument("--headless", action="store_true", help="no window: run for --seconds and print a summary")
    ap.add_argument("--seconds", type=int, default=30)
    args = ap.parse_args()

    yolo = Yolo(args.model, args.labels, args.conf, args.iou, args.threads)
    print(f"Model input {yolo.size_w}x{yolo.size_h}, {len(yolo.labels)} classes. Source {args.source}")
    print("Connecting to the camera ...")
    try:
        source = Source(args.source, args.baud)
    except OSError as exc:  # network errors and pyserial's SerialException are both OSError
        if args.source.startswith("tcp:"):
            _, host, port = args.source.split(":")
            print("\n" + connection_help.tcp_connect_message(exc, host, int(port)), file=sys.stderr)
        else:
            print("\n" + connection_help.serial_open_message(exc, args.source), file=sys.stderr)
        sys.exit(1)
    grabber = FrameGrabber(source)
    grabber.start()

    capture_dir = ROOT / "ESP_CAM" / "captures"
    seen = 0
    shown = None
    window_opened = False
    warned_no_frames = False
    infer_times, class_counter, detect_frames = [], Counter(), 0
    t_start = last_report = time.perf_counter()
    last_seen_count, yolo_fps, last_grab, grab_fps, infer_ms = 0, 0.0, 0, 0.0, 0.0
    try:
        while grabber.running:
            jpeg, count = grabber.latest()
            if jpeg is None or count == seen:  # wait for a new frame
                if cv2.waitKey(1) & 0xFF == ord("q") and not args.headless:
                    break
                if window_opened and window_closed(WINDOW):  # closed with the X button
                    break
                if count == 0 and not warned_no_frames and time.perf_counter() - t_start > 8:
                    print("\n" + connection_help.NO_FRAMES_MESSAGE)
                    warned_no_frames = True
                time.sleep(0.002)
                if args.headless and time.perf_counter() - t_start > args.seconds:
                    break
                continue
            seen = count
            frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if ROTATIONS[args.rotate] is not None:
                frame = cv2.rotate(frame, ROTATIONS[args.rotate])
            t0 = time.perf_counter()
            dets = yolo.detect(frame)
            infer_ms = (time.perf_counter() - t0) * 1000
            infer_times.append(infer_ms)
            if dets:
                detect_frames += 1
            for d in dets:
                class_counter[d[0]] += 1

            now = time.perf_counter()
            if now - last_report >= 1:
                yolo_fps = (len(infer_times) - last_seen_count) / (now - last_report)
                grab_fps = (count - last_grab) / (now - last_report)
                last_seen_count, last_grab, last_report = len(infer_times), count, now
                print(f"YOLO {yolo_fps:.1f}/s | infer {infer_ms:.0f} ms | camera {grab_fps:.1f} FPS | "
                      f"{', '.join(f'{d[0]} {int(d[1] * 100)}%' for d in dets[:4]) or 'nothing detected'}")
            if not args.headless:
                shown = frame.copy()
                draw(shown, dets, yolo_fps, infer_ms, grab_fps)
                cv2.imshow(WINDOW, shown)
                window_opened = True
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or window_closed(WINDOW):  # q key or the X button
                    break
                if key == ord("s"):
                    capture_dir.mkdir(exist_ok=True)
                    path = capture_dir / f"yolo_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(str(path), shown)
                    print(f"Saved {path}")
            elif now - t_start > args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        grabber.running = False
        cv2.destroyAllWindows()

    total = time.perf_counter() - t_start
    n = len(infer_times)
    print("\n=== summary ===")
    if grabber.error:
        print(connection_help.connection_lost_message(grabber.error, args.source.startswith("tcp:")))
    if n:
        print(f"frames processed {n} in {total:.0f} s ({n / total:.1f}/s); camera delivered {grabber.count}, bad {grabber.bad}")
        print(f"inference: mean {np.mean(infer_times):.0f} ms, median {np.median(infer_times):.0f} ms, p95 {np.percentile(infer_times, 95):.0f} ms")
        print(f"frames with at least one detection: {detect_frames} ({100 * detect_frames / n:.0f}%)")
        print("detections by class:", dict(class_counter.most_common(8)) or "none")
    else:
        print("no frames received")


if __name__ == "__main__":
    main()
