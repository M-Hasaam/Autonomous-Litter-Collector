"""Detect paper balls with the temporary 'yolo-temp' model and estimate how far away they are (camera only).

TEMPORARY test tool - it does not touch the app's model. The model is Training/paperball_yolo_temp/yolo-temp.onnx
(build it with Training/paperball_yolo_temp/prepare_dataset.py + train_yolo_temp.py).

Two distance estimates are made from each detected box:
  * floor  : the lower the ball's bottom edge sits in the image, the closer it is (needs --calibrate once).
             Does not depend on the ball's size, so a big ball and a small ball at the same spot give the same value.
  * size   : the smaller the box, the farther the ball (assumes the ball is --ball-cm wide; calibration refines it).
With a calibration the floor value is shown and the box width is turned into a size ratio, so a ball that is much
bigger than the calibration ball is flagged "BIG" instead of being picked. Without one, only the rough size value is
shown, marked "~" and "uncalibrated".

Usage:
  py camera_distance.py                                  live, through the bridge (tcp:192.168.4.1:8080)
  py camera_distance.py --source COM11                   live over a USB cable
  py camera_distance.py --calibrate                      place the ball at known distances, press SPACE each time
  py camera_distance.py --image a.jpg b.jpg --save-dir out   offline test on saved pictures
Keys in the window: q = quit, s = save the clean (no drawing) frame into Embedded/esp32-cam/captures/ for training.
"""
import argparse
import json
import math
import statistics
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

import connection_help
from camera_yolo import FrameGrabber, ROTATIONS, Source, window_closed

ROOT = Path(__file__).resolve().parents[4]
TEMP_DIR = ROOT / "Training" / "paperball_yolo_temp"
DEFAULT_MODEL = TEMP_DIR / "yolo-temp.onnx"
CALIB_FILE = TEMP_DIR / "distance_calibration.json"
CAPTURE_DIR = ROOT / "Embedded" / "esp32-cam" / "captures"
WINDOW = "ESP32-CAM paper-ball distance (yolo-temp)"
BIG_RATIO = 1.4  # a ball this much wider than the calibration ball (at the same distance) is "BIG"
GREEN, ORANGE, RED, YELLOW = (0, 255, 0), (0, 165, 255), (0, 0, 255), (0, 255, 255)


# ---------------------------------------------------------------- detector (ONNX through OpenCV, no extra packages)
class OnnxDetector:
    def __init__(self, path, size, conf, iou):
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.size, self.conf, self.iou = size, conf, iou

    def detect(self, frame):
        """List of (score, x1, y1, x2, y2) in frame pixels, best first. Same letterbox as the app (gray 114)."""
        h, w = frame.shape[:2]
        scale = min(self.size / w, self.size / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        pad_x, pad_y = (self.size - nw) / 2.0, (self.size - nh) / 2.0
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        x0, y0 = int(round(pad_x)), int(round(pad_y))
        canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        self.net.setInput(cv2.dnn.blobFromImage(canvas, 1 / 255.0, (self.size, self.size), swapRB=True, crop=False))
        out = self.net.forward()
        out = out.reshape(out.shape[-2], out.shape[-1])
        if out.shape[0] > out.shape[1]:
            out = out.T  # -> [4 + classes, anchors]
        score = out[4:].max(axis=0)
        keep = score > self.conf
        if not keep.any():
            return []
        cx, cy, bw, bh, score = out[0][keep], out[1][keep], out[2][keep], out[3][keep], score[keep]
        left = np.clip((cx - bw / 2 - pad_x) / scale, 0, w)
        top = np.clip((cy - bh / 2 - pad_y) / scale, 0, h)
        right = np.clip((cx + bw / 2 - pad_x) / scale, 0, w)
        bottom = np.clip((cy + bh / 2 - pad_y) / scale, 0, h)
        boxes = [[float(left[i]), float(top[i]), float(right[i] - left[i]), float(bottom[i] - top[i])]
                 for i in range(len(score))]
        dets = []
        for k in np.array(cv2.dnn.NMSBoxes(boxes, score.tolist(), self.conf, self.iou)).flatten():
            k = int(k)
            if right[k] > left[k] and bottom[k] > top[k]:
                dets.append((float(score[k]), int(left[k]), int(top[k]), int(right[k]), int(bottom[k])))
        return sorted(dets, key=lambda d: -d[0])


# ---------------------------------------------------------------- distance model
class DistanceModel:
    """Distances are in cm, measured from the camera lens to the FRONT edge of the ball along the floor."""

    def __init__(self, fov_deg, ball_cm, size_k=None, floor_ab=None, calibrated=False):
        self.fov_deg, self.ball_cm, self.calibrated = fov_deg, ball_cm, calibrated
        self.floor_ab = floor_ab  # 1/d = a*y_bottom + b
        self.size_k = size_k      # width_px * distance_to_ball_centre_cm  (= focal_px * ball width)

    def focal_px(self, frame_w, frame_h):
        return (max(frame_w, frame_h) / 2) / math.tan(math.radians(self.fov_deg) / 2)

    def estimate(self, box, frame_w, frame_h):
        _, x1, y1, x2, y2 = box
        width = max(x2 - x1, 1)
        radius = self.ball_cm / 2
        f = self.focal_px(frame_w, frame_h)
        k = self.size_k if self.size_k else f * self.ball_cm
        d_size = k / width - radius
        d_floor = None
        if self.floor_ab:
            v = self.floor_ab[0] * y2 + self.floor_ab[1]
            if v > 1e-5:
                d_floor = 1.0 / v
        ratio = None
        if d_floor is not None and self.size_k:
            ratio = width * (d_floor + radius) / self.size_k  # 1.0 = same size as the calibration ball
        return {
            "distance": d_floor if d_floor is not None else d_size,
            "floor": d_floor, "size": d_size, "ratio": ratio,
            "ball_cm": self.ball_cm * ratio if ratio else None,
            "bearing": math.degrees(math.atan(((x1 + x2) / 2 - frame_w / 2) / f)),
        }

    @classmethod
    def load(cls, fov_deg, ball_cm, rotate):
        if CALIB_FILE.is_file():
            data = json.loads(CALIB_FILE.read_text())
            if data.get("rotate") != rotate:
                print(f"WARNING: the calibration was made with --rotate {data.get('rotate')}, "
                      f"but now --rotate {rotate} is used. Re-run --calibrate.")
            return cls(data.get("fov_deg", fov_deg), data.get("ball_cm", ball_cm), data["size_k"],
                       (data["floor_a"], data["floor_b"]), calibrated=True)
        return cls(fov_deg, ball_cm)


def fit_calibration(samples, ball_cm, fov_deg, rotate):
    """samples: [(distance_cm, bottom_y_px, width_px)] -> calibration dict (and a printable quality report)."""
    d = np.array([s[0] for s in samples], float)
    y = np.array([s[1] for s in samples], float)
    w = np.array([s[2] for s in samples], float)
    if len(samples) < 3:
        raise ValueError("need at least 3 distances to calibrate")
    a, b = np.polyfit(y, 1.0 / d, 1)
    if a <= 0:
        raise ValueError("the bottom edge did not move down as the ball came closer; check the distances")
    fitted = 1.0 / (a * y + b)
    k = float(np.mean(w * (d + ball_cm / 2)))
    return {"floor_a": float(a), "floor_b": float(b), "size_k": k, "ball_cm": ball_cm, "fov_deg": fov_deg,
            "rotate": rotate, "points": [list(map(float, s)) for s in samples]}, fitted


# ---------------------------------------------------------------- drawing
def annotate(frame, dets, model, note=""):
    shown = frame.copy()
    h, w = shown.shape[:2]
    results = [(d, model.estimate(d, w, h)) for d in dets]
    target = max(results, key=lambda r: r[0][4])[1] if results else None  # nearest = lowest in the picture
    for det, est in results:
        _, x1, y1, x2, y2 = det
        big = est["ratio"] is not None and est["ratio"] > BIG_RATIO
        color = RED if big else (GREEN if est is target else ORANGE)
        cv2.rectangle(shown, (x1, y1), (x2, y2), color, 2)
        cv2.putText(shown, f"ball {int(det[0] * 100)}%", (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        mark = "" if model.calibrated else "approx "  # a "~" looks like a minus sign in this font
        lines = ["too close" if est["distance"] < 3 else f"{mark}{est['distance']:.0f} cm"]
        if est["ratio"] is not None:
            lines.append(f"BIG x{est['ratio']:.1f}" if big else f"size x{est['ratio']:.1f}")
        lines.append(f"{est['bearing']:+.0f} deg")
        for i, text in enumerate(lines):
            cv2.putText(shown, text, (x1, min(h - 6, y2 + 18 + 18 * i)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    status = "calibrated (floor + size)" if model.calibrated else "UNCALIBRATED: rough size-based guess"
    cv2.putText(shown, status + (f"  {note}" if note else ""), (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                GREEN if model.calibrated else YELLOW, 1)
    return shown, results


# ---------------------------------------------------------------- modes
def run_images(args, detector, model):
    out_dir = Path(args.save_dir) if args.save_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.image:
        frame = cv2.imread(path)
        if frame is None:
            print(f"{path}: cannot read")
            continue
        if ROTATIONS[args.rotate] is not None:
            frame = cv2.rotate(frame, ROTATIONS[args.rotate])
        dets = detector.detect(frame)
        shown, results = annotate(frame, dets, model)
        parts = []
        for det, est in results:
            floor = "-" if est["floor"] is None else f"{est['floor']:.0f}"
            parts.append(f"{int(det[0] * 100)}% -> {est['distance']:.0f} cm (floor {floor}, size {est['size']:.0f}), "
                         f"{est['bearing']:+.0f} deg")
        print(f"{Path(path).name}: {', '.join(parts) or 'no ball found'}")
        if out_dir:
            cv2.imwrite(str(out_dir / Path(path).name), shown)


def open_camera(args):
    print("Connecting to the camera ...")
    try:
        source = Source(args.source, args.baud)
    except OSError as exc:
        if args.source.startswith("tcp:"):
            _, host, port = args.source.split(":")
            print("\n" + connection_help.tcp_connect_message(exc, host, int(port)), file=sys.stderr)
        else:
            print("\n" + connection_help.serial_open_message(exc, args.source), file=sys.stderr)
        sys.exit(1)
    grabber = FrameGrabber(source)
    grabber.start()
    return grabber


def next_frame(grabber, seen, rotate):
    """(frame, count) for a new camera frame, or (None, seen) if there is none yet."""
    jpeg, count = grabber.latest()
    if jpeg is None or count == seen:
        return None, seen
    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if frame is not None and ROTATIONS[rotate] is not None:
        frame = cv2.rotate(frame, ROTATIONS[rotate])
    return frame, count


def run_live(args, detector, model):
    grabber = open_camera(args)
    seen, opened, t_start, warned = 0, False, time.perf_counter(), False
    history, last_print = deque(maxlen=5), 0.0
    try:
        while grabber.running:
            frame, seen_new = next_frame(grabber, seen, args.rotate)
            if frame is None:
                if cv2.waitKey(1) & 0xFF == ord("q") or (opened and window_closed(WINDOW)):
                    break
                if seen == 0 and not warned and time.perf_counter() - t_start > 8:
                    print("\n" + connection_help.NO_FRAMES_MESSAGE)
                    warned = True
                time.sleep(0.002)
                continue
            seen = seen_new
            dets = detector.detect(frame)
            shown, results = annotate(frame, dets, model)
            if results:
                nearest = max(results, key=lambda r: r[0][4])[1]
                history.append(nearest["distance"])
                now = time.perf_counter()
                if now - last_print >= 1:
                    last_print = now
                    flag = "" if model.calibrated else " (uncalibrated)"
                    print(f"nearest ball: {statistics.median(history):.0f} cm{flag}, {nearest['bearing']:+.0f} deg, "
                          f"{len(results)} ball(s)")
            else:
                history.clear()
            cv2.imshow(WINDOW, shown)
            opened = True
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or window_closed(WINDOW):
                break
            if key == ord("s"):
                CAPTURE_DIR.mkdir(exist_ok=True)
                path = CAPTURE_DIR / f"frame_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(str(path), frame)  # clean frame, no boxes drawn: good for training
                print(f"Saved {path}")
    except KeyboardInterrupt:
        pass
    finally:
        grabber.running = False
        cv2.destroyAllWindows()
    if grabber.error:
        print("\n" + connection_help.connection_lost_message(grabber.error, args.source.startswith("tcp:")),
              file=sys.stderr)


def run_calibrate(args, detector, model):
    distances = [float(v) for v in args.calib_distances.split(",")]
    print(f"Calibration: for each distance put ONE ball on the floor, in the middle of the picture, with its FRONT "
          f"edge that far from the camera lens (measure along the floor), then press SPACE.\n"
          f"Distances: {distances} cm. Keys: SPACE = record, N = skip this distance, Q = quit.")
    grabber = open_camera(args)
    samples, index, seen, opened = [], 0, 0, False
    recording, taken, started = False, [], 0.0
    try:
        while grabber.running and index < len(distances):
            frame, seen_new = next_frame(grabber, seen, args.rotate)
            if frame is None:
                if cv2.waitKey(1) & 0xFF == ord("q") or (opened and window_closed(WINDOW)):
                    break
                time.sleep(0.002)
                continue
            seen = seen_new
            dets = detector.detect(frame)
            shown, _ = annotate(frame, dets, model)
            if recording:
                if len(dets) == 1:
                    taken.append(dets[0])
                if len(taken) >= 15 or time.perf_counter() - started > 6:
                    recording = False
                    if len(taken) >= 5:
                        y = statistics.median(t[4] for t in taken)
                        w = statistics.median(t[3] - t[1] for t in taken)
                        samples.append((distances[index], y, w))
                        print(f"  {distances[index]:.0f} cm: bottom y {y:.0f}, width {w:.0f} px  ({len(taken)} frames)")
                        index += 1
                    else:
                        print("  the ball was not seen clearly (need exactly one ball in view); try again")
            msg = (f"RECORDING ({len(taken)}/15) ..." if recording else
                   f"Place the ball at {distances[index]:.0f} cm - SPACE to record, N to skip")
            cv2.putText(shown, msg, (6, shown.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 2)
            cv2.imshow(WINDOW, shown)
            opened = True
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or window_closed(WINDOW):
                print("Calibration cancelled.")
                return
            if key == ord(" ") and not recording:
                recording, taken, started = True, [], time.perf_counter()
            if key == ord("n") and not recording:
                index += 1
    except KeyboardInterrupt:
        return
    finally:
        grabber.running = False
        cv2.destroyAllWindows()
    if grabber.error:
        print("\n" + connection_help.connection_lost_message(grabber.error, args.source.startswith("tcp:")),
              file=sys.stderr)
        return
    try:
        calib, fitted = fit_calibration(samples, args.ball_cm, args.fov, args.rotate)
    except ValueError as exc:
        print(f"Calibration failed: {exc}")
        return
    print("\ndistance  fitted   error")
    for (d, _, _), f in zip(samples, fitted):
        print(f"{d:7.0f}  {f:6.1f}  {f - d:+6.1f} cm")
    CALIB_FILE.write_text(json.dumps(calib, indent=2))
    print(f"\nSaved {CALIB_FILE}. Run without --calibrate to use it.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="tcp:192.168.4.1:8080", help="tcp:HOST:PORT (bridge) or a COM port")
    ap.add_argument("--baud", type=int, default=921_600, help="only for a COM port source")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.5, help="confidence threshold (the tiny temp model needs it high)")
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--rotate", choices=ROTATIONS, default="none", help="same meaning as camera_yolo.py")
    ap.add_argument("--ball-cm", type=float, default=7.0, help="width of the paper ball in cm (default 7)")
    ap.add_argument("--fov", type=float, default=55.0,
                    help="camera field of view in degrees across the longest side of the picture (default 55)")
    ap.add_argument("--calibrate", action="store_true", help="record known distances and save the calibration")
    ap.add_argument("--calib-distances", default="20,30,40,50,60,80", help="distances in cm to record")
    ap.add_argument("--image", nargs="+", help="test on saved pictures instead of the camera")
    ap.add_argument("--save-dir", help="with --image: write the annotated pictures here")
    args = ap.parse_args()

    if not Path(args.model).is_file():
        print(f"Model not found: {args.model}\nBuild it first: py Training/paperball_yolo_temp/prepare_dataset.py, then "
              f"train_yolo_temp.py in an Ultralytics environment.", file=sys.stderr)
        sys.exit(1)
    detector = OnnxDetector(args.model, args.imgsz, args.conf, args.iou)
    model = DistanceModel.load(args.fov, args.ball_cm, args.rotate)
    if args.image:
        run_images(args, detector, model)
    elif args.calibrate:
        run_calibrate(args, detector, DistanceModel(args.fov, args.ball_cm))
    else:
        run_live(args, detector, model)


if __name__ == "__main__":
    main()
