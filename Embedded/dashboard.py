"""LitterVision car dashboard: camera + sensors + drive + servos in one window.

  * Camera   - ESP32 WiFi bridge, TCP 192.168.4.1:8080 (join the "ESP32-BRIDGE" WiFi first).
  * Car      - ESP32-S3 "AI_Car" over Bluetooth LE (drive, servos, distance, MPU-6050).

Both links connect (and reconnect) on their own, so start the dashboard first and power things on later.

Install:  pip install bleak opencv-python pillow numpy
Run:      py dashboard.py
Keys:     Arrow keys / W A S D drive (release to stop), Esc quits. One speed slider sets all wheels.
"""
import asyncio
import math
import queue
import socket
import struct
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import simpledialog, ttk

import cv2
import numpy as np
from bleak import BleakClient, BleakScanner
from PIL import Image, ImageTk

# ---------------------------------------------------------------- settings
# Camera (ESP32 WiFi bridge)
CAMERA_HOST = "192.168.4.1"
CAMERA_PORT = 8080
CAMERA_WIFI_NAME = "ESP32-BRIDGE"
CAMERA_ROTATE = "none"            # none | cw | ccw | 180   (also selectable in the UI)
CAPTURE_DIR = Path(__file__).resolve().parent / "captures"

# Car (ESP32-S3 over BLE)
CAR_NAME = "AI_Car"
CHAR_UUID = "87654321-4321-4321-4321-BA0987654321"           # commands (write)
DISTANCE_CHAR_UUID = "11223344-5566-7788-99AA-BBCCDDEEFF00"  # distance cm, "-1" = no echo
MPU_CHAR_UUID = "22334455-6677-8899-AABB-CCDDEEFF0011"       # "yaw,pitch,roll,gx,gy,gz,ax,ay,az"

SERVO_NAMES = ["Base Rotation", "Shoulder", "Elbow", "Gripper"]
SERVO_SPEED = 60                  # default deg/s
CAR_SPEED_DEFAULT = 100           # wheel speed in percent (one slider for all wheels)
CAR_SPEED_MIN = 20                # below this the wheels usually stall

# Path map (dead reckoning from gyro + motor commands, no encoders)
CM_PER_S_AT_100 = 30.0            # how fast the car drives at 100% wheel speed - calibrate with "Test drive"
REPLAY_TURN_PCT = 70              # wheel speed for the first, fast part of a gyro-controlled turn
REPLAY_FINE_PCT = 55              # wheel speed for the last ~30 degrees of a turn (slower = more accurate)
                                  # (if the wheels only hum, the dashboard raises these by itself - stall detection)
BACK_SPEED_RATIO = 1.0            # reverse speed / forward speed (measure it: a 10% gap = ~5% error on the way home)
TEST_DRIVE_S = 3.0                # length of the calibration drive
SERVO_SEND_INTERVAL = 0.06        # seconds between live commands while dragging

# Flip to -1 if a direction shows backwards for how the MPU is mounted.
YAW_SIGN = 1                      # +1: turning the car left gives a positive heading
PITCH_SIGN = 1                    # +1: positive = front tilted up
ROLL_SIGN = 1                     # +1: positive = leaning right

DIST_WARN_CM = 50                 # amber below this
DIST_STOP_CM = 20                 # red below this

# ---------------------------------------------------------------- theme
BG = "#0d1117"
CARD = "#161b22"
CARD_HI = "#21262d"
LINE = "#30363d"
TEXT = "#e6edf3"
MUTED = "#8b949e"
ACCENT = "#2f81f7"
GOOD = "#3fb950"
WARN = "#d29922"
BAD = "#f85149"
FONT = "Segoe UI"

ROTATIONS = {"none": None, "cw": cv2.ROTATE_90_CLOCKWISE, "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE, "180": cv2.ROTATE_180}
MAGIC = b"\xa5\x5a\xc3\x3c"       # frame = MAGIC | uint32 LE length | JPEG
MAX_FRAME = 300_000


# ---------------------------------------------------------------- camera
def extract_frames(buf):
    """Pull every complete JPEG out of buf (bytearray, modified in place). Returns (jpegs, bad_count)."""
    jpegs, bad = [], 0
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
        jpegs.append(jpeg)
    return jpegs, bad


class CameraWorker(threading.Thread):
    """Connects to the bridge, keeps the newest decoded frame, reconnects forever."""

    def __init__(self, host, port):
        super().__init__(daemon=True)
        self.host, self.port = host, port
        self.stop_event = threading.Event()
        self.rotate = ROTATIONS.get(CAMERA_ROTATE)
        self.lock = threading.Lock()
        self.frame_rgb = None        # newest frame (RGB ndarray)
        self.frame_id = 0
        self.last_jpeg = None        # camera's exact JPEG, for saving
        self.fps = 0.0
        self.link_kbps = 0.0         # WiFi throughput actually received (KB/s)
        self.connected = False
        self.message = "Starting..."
        self.silent_hint = ""        # set when the bridge answered but sent nothing
        self._wifi_checked = 0.0
        self._wifi_ssid = None

    def snapshot(self):
        with self.lock:
            return self.frame_id, self.frame_rgb

    @staticmethod
    def _read_wifi_ssid():
        """Windows WiFi network name; '' if disconnected; None if it can't be read."""
        try:
            out = subprocess.run(["netsh", "wlan", "show", "interfaces"], capture_output=True, text=True,
                                 timeout=4, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        except Exception:
            return None
        state = ssid = ""
        for line in out.splitlines():
            key, _, value = line.partition(":")
            key, value = key.strip().lower(), value.strip()
            if key == "state":
                state = value.lower()
            elif key == "ssid":
                ssid = value
        if not state:
            return None
        return ssid if state == "connected" else ""

    def _waiting_message(self):
        """Say what we're waiting for, based on which WiFi the PC is on (re-checked every 3 s)."""
        if time.monotonic() - self._wifi_checked > 3:
            self._wifi_checked = time.monotonic()
            self._wifi_ssid = self._read_wifi_ssid()
        ssid = self._wifi_ssid
        if ssid is None:
            return f"No camera - waiting for {self.host} (retrying)"
        if ssid == "":
            return f"WiFi not connected - join '{CAMERA_WIFI_NAME}' (retrying)"
        if ssid != CAMERA_WIFI_NAME:
            return f"On WiFi '{ssid}' - join '{CAMERA_WIFI_NAME}' (retrying)"
        return f"On {CAMERA_WIFI_NAME} - waiting for the bridge to answer (retrying)"

    def run(self):
        while not self.stop_event.is_set():
            try:
                if not self.silent_hint:
                    self.message = f"Connecting to {self.host}:{self.port} ..."
                sock = socket.create_connection((self.host, self.port), timeout=2)
            except OSError:
                self.connected = False
                self.fps = 0.0
                self.message = self.silent_hint or self._waiting_message()
                self.stop_event.wait(1.0)  # retry every second: picks the bridge up as soon as it appears
                continue
            self._stream(sock)
            self.connected = False
            self.fps = 0.0
            self.stop_event.wait(0.5)

    def _stream(self, sock):
        sock.settimeout(0.5)
        self.connected = True
        self.message = "Waiting for frames..."
        buf = bytearray()
        last_data = time.monotonic()
        count, mark = 0, time.monotonic()
        nbytes = 0
        try:
            while not self.stop_event.is_set():
                try:
                    data = sock.recv(65536)
                    if not data:
                        break
                    buf.extend(data)
                    nbytes += len(data)
                    last_data = time.monotonic()
                except socket.timeout:
                    if time.monotonic() - last_data > 6:
                        # The bridge accepts only ONE viewer at a time, and sends nothing if the camera is off.
                        self.silent_hint = ("Bridge connected but no frames - close other camera viewers "
                                            "(camera_yolo.py etc.) and check the camera")
                        self.message = self.silent_hint
                        break
                    continue
                if len(buf) > 2_000_000:
                    del buf[:-200_000]
                jpegs, _bad = extract_frames(buf)
                if not jpegs:
                    continue
                jpeg = jpegs[-1]  # only the newest matters
                img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                rot = self.rotate
                if rot is not None:
                    img = cv2.rotate(img, rot)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                with self.lock:
                    self.frame_rgb = rgb
                    self.last_jpeg = jpeg
                    self.frame_id += 1
                count += len(jpegs)
                self.silent_hint = ""
                self.message = "Live"
                now = time.monotonic()
                if now - mark >= 1:
                    self.fps = count / (now - mark)
                    self.link_kbps = nbytes / 1000 / (now - mark)
                    count, nbytes, mark = 0, 0, now
        except OSError:
            self.message = "Camera connection lost - reconnecting"
        finally:
            sock.close()


# ---------------------------------------------------------------- dead reckoning (no encoders)
def wrap180(angle):
    """Wrap an angle in degrees into [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


class PathTracker:
    """Estimates where the car is: heading from the gyro, distance from drive time x speed.

    Map frame: the car starts at (0, 0) facing +y ("up"). Heading is in degrees, + = turning left (counter-clockwise).
    `corners` is the recorded route: the origin plus the point where every straight drive ended. Driving back through
    them in reverse order retraces the path.
    """
    MIN_MOVE_CM = 2.0
    CORNER_MERGE_CM = 3.0
    POINT_SPACING_CM = 1.0
    MAX_POINTS = 20000
    MAX_EXTRAPOLATE_S = 0.2

    def __init__(self, cm_per_s_100=CM_PER_S_AT_100):
        self.cm_per_s_100 = cm_per_s_100
        self.recording = True
        self.yaw = 0.0                 # unwrapped gyro yaw, degrees
        self.gz = 0.0                  # latest turning speed (deg/s), used to predict the heading between readings
        self.yaw_time = 0.0
        self._last_wrapped = None
        self.cmd = "S"
        self.reset()

    @property
    def heading(self):
        return self.yaw - self.yaw_ref

    def heading_now(self, now):
        """Heading predicted for time `now`: the gyro reading plus how far we have turned since it was taken."""
        age = min(max(now - self.yaw_time, 0.0), self.MAX_EXTRAPOLATE_S)
        return self.heading + self.gz * age

    def reset(self):
        """Forget the path: here becomes (0, 0), and the way the car faces becomes heading 0."""
        self.yaw_ref = self.yaw
        self.x = self.y = 0.0
        self.total_cm = 0.0
        self.points = [(0.0, 0.0, "out")]
        self.corners = [(0.0, 0.0)]
        self.phase = "out"             # "out" while exploring, "back" while returning
        self.cmd = "S"
        self._moving, self._val = False, 0.0

    def feed_yaw(self, wrapped, resync=False, gz=0.0, now=None):
        """New gyro heading from the car (wrapped to -180..180). resync = the car's yaw was zeroed or it rebooted."""
        if self._last_wrapped is None or resync:
            keep = 0.0 if self._last_wrapped is None else self.heading
            self.yaw = wrapped
            self.yaw_ref = self.yaw - keep       # the map heading does not jump
        else:
            self.yaw += wrap180(wrapped - self._last_wrapped)
        self._last_wrapped = wrapped
        self.gz = gz
        self.yaw_time = time.monotonic() if now is None else now

    def set_command(self, cmd):
        """Call whenever a drive command (U D L R S) is sent to the car."""
        if cmd == self.cmd:
            return
        self._close_segment()
        self.cmd = cmd
        self._moving, self._val = cmd in ("U", "D"), 0.0

    def _close_segment(self):
        if self.recording and self._moving and abs(self._val) >= self.MIN_MOVE_CM:
            lx, ly = self.corners[-1]
            if math.hypot(self.x - lx, self.y - ly) >= self.CORNER_MERGE_CM:
                self.corners.append((self.x, self.y))
        self._moving = False

    def update(self, dt, speed_pct, now=None):
        """Advance the position estimate by dt seconds of whatever the car is currently doing."""
        if self.cmd not in ("U", "D") or dt <= 0:
            return
        d = self.cm_per_s_100 * speed_pct / 100.0 * dt * (1 if self.cmd == "U" else -BACK_SPEED_RATIO)
        h = math.radians(self.heading if now is None else self.heading_now(now))
        self.x += -math.sin(h) * d
        self.y += math.cos(h) * d
        self._val += d
        self.total_cm += abs(d)
        lx, ly, _ = self.points[-1]
        if math.hypot(self.x - lx, self.y - ly) >= self.POINT_SPACING_CM:
            self.points.append((self.x, self.y, self.phase))
            if len(self.points) > self.MAX_POINTS:
                del self.points[: self.MAX_POINTS // 10]


def return_steps(corners):
    """Drive back through the recorded corners, newest first, always forward; then face the way we started."""
    steps = [["goto", x, y] for x, y in reversed(corners[:-1])]
    if steps:
        steps.append(["turn_to", 0.0])
    return steps


def plan_steps(waypoints):
    """Drive through the waypoints clicked on the map (map cm). Each leg is re-aimed from where the car really is."""
    return [["goto", x, y] for x, y in waypoints]


class PathRunner:
    """Drives a list of steps on its own, re-aiming before every leg.

    Steps: ["goto", x, y]          turn towards the point (gyro), then drive forward until the estimate reaches it
           ["turn_to", heading]    closed-loop in-place turn on the gyro
           ["move", cm, tx, ty]    (made from a goto) forward drive; tx/ty let it stop when the estimate arrives
    tick(now, gz) must be called every ~30 ms; send(cmd) sends a drive command to the car.
    """
    TURN_TOL_DEG = 2.0
    LATENCY_S = 0.20            # BLE + motor lag: release the turn this early (scaled by the turning speed)
    SETTLE_S = 0.45             # wait for the car to stop before judging a turn
    FINE_ZONE_DEG = 30.0        # inside this error the turn slows down to REPLAY_FINE_PCT
    MAX_CORRECTIONS = 3
    TURN_TIMEOUT_S = 15.0
    GOTO_TOL_CM = 3.0
    STALL_RATE_DPS = 4.0        # turning slower than this while a turn command is active = the wheels are stalled
    STALL_S = 0.6               # ... for this long (longer than the Bluetooth + motor lag)
    STALL_STEP_PCT = 10         # raise the turn speed by this much each time

    def __init__(self, tracker, send, get_speed, get_dist, on_finish=None):
        self.tracker, self.send = tracker, send
        self.get_speed, self.get_dist = get_speed, get_dist
        self.on_finish = on_finish
        self.active = False
        self.status = "Idle"
        self.label = ""
        self.steps, self.idx, self.state = [], 0, "idle"
        self.legs_done = self.legs_total = 0
        self._speed_changed = False
        self.last_turn_error = None
        self.floor_pct = 0              # lowest turn speed at which the wheels were seen to actually turn
        self.cur_pct = REPLAY_TURN_PCT
        self.cmd_since = 0.0

    def _turn_pct(self, fine):
        return max(REPLAY_FINE_PCT if fine else REPLAY_TURN_PCT, self.floor_pct)

    def start(self, steps, label, now):
        if not steps:
            self.status = f"{label}: nothing to do"
            return False
        self.steps = [list(s) for s in steps]
        self.idx, self.label, self.state = 0, label, "next"
        self.legs_done, self.legs_total = 0, sum(1 for s in steps if s[0] == "goto")
        self.last_turn_error = None
        self.active = True
        self.status = f"{label}: starting"
        return True

    def abort(self, reason):
        if not self.active:
            return
        self.active, self.state = False, "idle"
        self.send("S")
        self._restore_speed()
        self.status = f"{self.label} stopped: {reason}"
        if self.on_finish:
            self.on_finish(False)

    def _finish(self):
        self.active, self.state = False, "idle"
        self.send("S")
        self._restore_speed()
        extra = f" (turn error {self.last_turn_error:+.1f}°)" if self.last_turn_error is not None else ""
        self.status = f"{self.label}: done{extra}"
        if self.on_finish:
            self.on_finish(True)

    def _set_speed(self, pct):
        self.send(f"P{pct}")
        self._speed_changed = pct != self.get_speed()

    def _restore_speed(self):
        if self._speed_changed:
            self._speed_changed = False
            self.send(f"P{self.get_speed()}")

    def tick(self, now, gz):
        if not self.active:
            return
        tr, tol = self.tracker, self.TURN_TOL_DEG
        if self.state == "next":
            if self.idx >= len(self.steps):
                self._finish()
                return
            step = self.steps[self.idx]
            if step[0] == "goto":                      # re-aim from where the car is now
                dx, dy = step[1] - tr.x, step[2] - tr.y
                dist = math.hypot(dx, dy)
                self.legs_done += 1
                if dist < self.GOTO_TOL_CM:
                    self.idx += 1
                    return
                bearing = math.degrees(math.atan2(-dx, dy))      # 0 = up, + = left
                self.steps[self.idx:self.idx + 1] = [["turn_to", bearing], ["move", dist, step[1], step[2]]]
                self.status = f"{self.label}: leg {self.legs_done}/{self.legs_total}"
                return
            if step[0] == "move":
                v = tr.cm_per_s_100 * self.get_speed() / 100.0 * (1.0 if step[1] > 0 else BACK_SPEED_RATIO)
                if v <= 0:
                    self.abort("speed is 0")
                    return
                self._restore_speed()
                self.send("U" if step[1] > 0 else "D")
                self.until = now + abs(step[1]) / v * (1.0 if len(step) < 4 else 1.6)
                self.state = "moving"
            else:                                      # turn_to
                err = wrap180(step[1] - tr.heading_now(now))
                if abs(err) <= tol:
                    self.last_turn_error = err
                    self.idx += 1
                    return
                self.cur_pct = self._turn_pct(False)
                self._set_speed(self.cur_pct)
                self.corrections, self.turn_started, self.fine = 0, now, False
                self.cmd_since = now
                self.state = "turning"
        elif self.state == "moving":
            step = self.steps[self.idx]
            if tr.cmd == "U":
                d = self.get_dist()
                if d is not None and d < DIST_STOP_CM:
                    self.abort("obstacle ahead")
                    return
            arrived = now >= self.until
            if len(step) >= 4 and step[1] > 0:         # stop when the estimate reaches the target point
                h = math.radians(tr.heading_now(now))
                if (step[2] - tr.x) * -math.sin(h) + (step[3] - tr.y) * math.cos(h) <= 0.5:
                    arrived = True
            if arrived:
                self.send("S")
                self.state, self.settle_until, self.after_settle = "settle", now + self.SETTLE_S, "advance"
        elif self.state == "turning":
            if now - self.turn_started > self.TURN_TIMEOUT_S:
                self.abort("turn timed out")
                return
            err = wrap180(self.steps[self.idx][1] - tr.heading_now(now))
            if abs(err) < self.FINE_ZONE_DEG and not self.fine:
                self.fine = True
                self.cur_pct = self._turn_pct(True)
                self._set_speed(self.cur_pct)          # slow down for the last part: less overshoot
                self.cmd_since = now
            if abs(err) <= max(tol, abs(gz) * self.LATENCY_S):
                self.send("S")
                self.state, self.settle_until, self.after_settle = "settle", now + self.SETTLE_S, "check"
            else:
                want = "L" if err > 0 else "R"
                if tr.cmd != want:
                    self.send(want)
                    self.cmd_since = now
                elif abs(gz) < self.STALL_RATE_DPS and now - self.cmd_since > self.STALL_S and self.cur_pct < 100:
                    # Commanded to turn but not turning: the wheels only hum. Give them more power, and remember it.
                    self.cur_pct = min(100, self.cur_pct + self.STALL_STEP_PCT)
                    self.floor_pct = self.cur_pct
                    self._set_speed(self.cur_pct)
                    self.cmd_since = now
                    self.status = f"{self.label}: wheels stalled, turn speed raised to {self.cur_pct}%"
        elif self.state == "settle":
            if now < self.settle_until:
                return
            if self.after_settle == "check":
                err = wrap180(self.steps[self.idx][1] - tr.heading_now(now))
                if abs(err) > tol * 1.5 and self.corrections < self.MAX_CORRECTIONS:
                    self.corrections += 1
                    self.state = "turning"
                    return
                self.last_turn_error = err
            self.idx += 1
            self.state = "next"


class PathWindow:
    """The map: estimated path (blue = going out, green = coming back), click to plan waypoints."""

    def __init__(self, app):
        self.app = app
        win = self.win = tk.Toplevel(app.root)
        win.title("Path map - gyro + motors only (no encoders)")
        win.configure(bg=BG)
        h = min(800, win.winfo_screenheight() - 110)
        win.geometry(f"660x{h}+80+20")
        win.minsize(520, 560)
        win.protocol("WM_DELETE_WINDOW", self.close)
        self.waypoints = []
        self.tf = None
        self._last_draw = 0.0

        self.canvas = tk.Canvas(win, bg="#0a0e14", highlightbackground=LINE, highlightthickness=1)
        self.canvas.pack(fill="both", expand=True, padx=12, pady=(12, 6))
        self.canvas.bind("<Button-1>", self._click)
        self.canvas.bind("<Button-3>", lambda e: self._undo_waypoint())
        self.canvas.bind("<Configure>", lambda e: self.refresh(force=True))

        info = tk.Frame(win, bg=BG)
        info.pack(fill="x", padx=12)
        self.pos_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="")
        tk.Label(info, textvariable=self.pos_var, bg=BG, fg=TEXT, font=(FONT, 10, "bold"), justify="left").pack(anchor="w")
        tk.Label(info, textvariable=self.status_var, bg=BG, fg=MUTED, font=(FONT, 10)).pack(anchor="w")
        tk.Label(info, text="Click the map to plan waypoints (right-click removes the last one). Drive by hand and the "
                            "path is recorded.", bg=BG, fg=MUTED, font=(FONT, 9), wraplength=620, justify="left").pack(anchor="w")

        row1 = tk.Frame(win, bg=BG)
        row1.pack(fill="x", padx=12, pady=(8, 0))
        app._button(row1, "Return to start", app.path_return).pack(side="left")
        app._button(row1, "Run planned path", lambda: app.path_run_plan(self.waypoints)).pack(side="left", padx=6)
        app._button(row1, "ABORT", app.path_abort, danger=True).pack(side="left")
        app._button(row1, "Clear path (here = origin)", app.path_clear).pack(side="right")
        app._button(row1, "Clear plan", self.clear_plan).pack(side="right", padx=6)

        row2 = tk.Frame(win, bg=BG)
        row2.pack(fill="x", padx=12, pady=(8, 12))
        tk.Label(row2, text="Car speed at 100% (cm/s)", bg=BG, fg=MUTED, font=(FONT, 9)).pack(side="left")
        self.speed_var = tk.StringVar(value=f"{app.tracker.cm_per_s_100:.1f}")
        entry = tk.Entry(row2, textvariable=self.speed_var, width=7, bg=CARD_HI, fg=TEXT, insertbackground=TEXT,
                         relief="flat", highlightthickness=1, highlightbackground=LINE)
        entry.pack(side="left", padx=6)
        entry.bind("<Return>", lambda e: self.apply_speed())
        entry.bind("<FocusOut>", lambda e: self.apply_speed())
        app._button(row2, f"Test drive {TEST_DRIVE_S:.0f} s (calibrate)", app.path_test_drive).pack(side="left", padx=6)

    def close(self):
        self.app.path_window = None
        self.win.destroy()

    def apply_speed(self):
        try:
            value = float(self.speed_var.get())
            if value > 0:
                self.app.tracker.cm_per_s_100 = value
        except ValueError:
            pass

    def clear_plan(self):
        self.waypoints = []
        self.refresh(force=True)

    def _undo_waypoint(self):
        if self.waypoints:
            self.waypoints.pop()
            self.refresh(force=True)

    def _click(self, event):
        if self.tf is None:
            return
        cx, cy, scale, w, h = self.tf
        self.waypoints.append((cx + (event.x - w / 2) / scale, cy - (event.y - h / 2) / scale))
        self.refresh(force=True)

    @staticmethod
    def _grid_step(span):
        for step in (5, 10, 20, 25, 50, 100, 200, 250, 500, 1000):
            if span / step <= 10:
                return step
        return 2000

    def refresh(self, force=False):
        now = time.monotonic()
        if not force and now - self._last_draw < 0.1:
            return
        self._last_draw = now
        tr, c = self.app.tracker, self.canvas
        w, h = max(c.winfo_width(), 50), max(c.winfo_height(), 50)
        pts = tr.points
        xs = [p[0] for p in pts] + [x for x, _ in self.waypoints] + [tr.x]
        ys = [p[1] for p in pts] + [y for _, y in self.waypoints] + [tr.y]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        half = max((max(max(xs) - min(xs), max(ys) - min(ys))) / 2 * 1.2, 60.0)
        scale = min(w, h) / 2 / half
        self.tf = (cx, cy, scale, w, h)

        def px(x, y):
            return w / 2 + (x - cx) * scale, h / 2 - (y - cy) * scale

        c.delete("all")
        # grid with cm labels
        step = self._grid_step(2 * half * w / min(w, h))
        gx = math.floor((cx - w / 2 / scale) / step) * step
        while gx <= cx + w / 2 / scale:
            x0, _ = px(gx, 0)
            c.create_line(x0, 0, x0, h, fill="#1c2430" if gx else "#2f3b4d")
            c.create_text(x0 + 3, h - 3, text=f"{gx:g}", anchor="sw", fill="#4d5b6e", font=(FONT, 7))
            gx += step
        gy = math.floor((cy - h / 2 / scale) / step) * step
        while gy <= cy + h / 2 / scale:
            _, y0 = px(0, gy)
            c.create_line(0, y0, w, y0, fill="#1c2430" if gy else "#2f3b4d")
            c.create_text(3, y0 - 2, text=f"{gy:g} cm", anchor="sw", fill="#4d5b6e", font=(FONT, 7))
            gy += step
        # plan (dashed grey) from the car through the waypoints
        if self.waypoints:
            line = [px(tr.x, tr.y)] + [px(x, y) for x, y in self.waypoints]
            c.create_line(*[v for p in line for v in p], fill="#8b949e", dash=(5, 4))
            for i, (x, y) in enumerate(self.waypoints, 1):
                X, Y = px(x, y)
                c.create_oval(X - 6, Y - 6, X + 6, Y + 6, outline="#d29922", width=2)
                c.create_text(X, Y - 14, text=str(i), fill="#d29922", font=(FONT, 9, "bold"))
        # travelled path, coloured by phase
        stride = max(1, len(pts) // 3000)
        shown = pts[::stride] + ([pts[-1]] if stride > 1 else [])
        run, colour = [], None
        for x, y, phase in shown:
            col = ACCENT if phase == "out" else GOOD
            if colour is not None and col != colour and len(run) >= 2:
                c.create_line(*run, fill=colour, width=3, capstyle="round", joinstyle="round")
                run = run[-2:]
            colour = col
            run += list(px(x, y))
        if len(run) >= 4:
            c.create_line(*run, fill=colour, width=3, capstyle="round", joinstyle="round")
        # start marker
        sx, sy = px(0, 0)
        c.create_oval(sx - 6, sy - 6, sx + 6, sy + 6, outline=TEXT, width=2)
        c.create_text(sx + 10, sy + 10, text="start", anchor="nw", fill=TEXT, font=(FONT, 8))
        # the car (triangle pointing the way it faces; heading + = left / counter-clockwise)
        hr = math.radians(tr.heading)
        fx, fy = -math.sin(hr), -math.cos(hr)      # forward, in pixels
        rx, ry = math.cos(hr), -math.sin(hr)       # right, in pixels
        X, Y = px(tr.x, tr.y)
        tip = (X + fx * 14, Y + fy * 14)
        bl = (X - fx * 8 - rx * 8, Y - fy * 8 - ry * 8)
        br = (X - fx * 8 + rx * 8, Y - fy * 8 + ry * 8)
        c.create_polygon(*tip, *bl, *br, fill=WARN if self.app.runner.active else "#f0f6fc", outline="")

        self.pos_var.set(f"Position  x {tr.x:+.0f} cm   y {tr.y:+.0f} cm      Heading {wrap180(tr.heading):+.0f}°\n"
                         f"Driven {tr.total_cm:.0f} cm in total      Corners recorded: {len(tr.corners) - 1}")
        self.status_var.set(self.app.runner.status)


# ---------------------------------------------------------------- dashboard
class Dashboard:
    def __init__(self, root):
        self.root = root
        root.title("LitterVision Dashboard")
        # Fit the screen (leave room for the title bar and taskbar).
        w = min(1320, root.winfo_screenwidth() - 40)
        h = min(880, root.winfo_screenheight() - 110)
        root.geometry(f"{w}x{h}+20+10")
        root.minsize(min(1120, w), min(740, h))
        root.configure(bg=BG)

        # BLE state
        self.loop = None
        self.client = None
        self.characteristic = None
        self.write_response = False
        self.connected = threading.Event()
        self.stop_event = threading.Event()
        self.closing = False
        self.active_key = None
        self.heading_deg = 0.0

        # UI-thread plumbing: worker threads post callables here, _pump() runs them.
        self.ui_queue = queue.Queue()

        self.car_text = tk.StringVar(value="Searching...")
        self.cam_text = tk.StringVar(value="Starting...")
        self.dist_var = tk.StringVar(value="--")
        self.dist_note = tk.StringVar(value="waiting for sensor")
        self.heading_var = tk.StringVar(value="--")
        self.spin_var = tk.StringVar(value="Turning speed: --")
        self.pitch_var = tk.StringVar(value="Front-back tilt: --")
        self.roll_var = tk.StringVar(value="Side tilt: --")
        self.rotate_var = tk.StringVar(value=CAMERA_ROTATE)
        self.run_status_var = tk.StringVar(value="")       # progress of turns / planned path / return
        self.photo = None
        self.last_frame_id = -1
        self.saved = 0

        # Path map: dead reckoning from the gyro heading + motor commands (no encoders)
        self.tracker = PathTracker()
        self.runner = PathRunner(self.tracker, self._runner_send, lambda: int(self.car_speed.get()),
                                 lambda: self.dist_cm, self._runner_finished)
        self.path_window = None
        self.dist_cm = None            # latest distance reading, None = nothing in range
        self.gz = 0.0                  # latest turning speed (deg/s)
        self._yaw_resync = True        # the car's yaw was zeroed / it rebooted: don't treat the jump as a turn
        self._runner_sending = False
        self._last_tick = time.monotonic()
        self._pressed_button = False

        self._build_ui()

        self.camera = CameraWorker(CAMERA_HOST, CAMERA_PORT)
        self.camera.start()
        if "--no-ble" in sys.argv:  # test switch: camera only, to see if Bluetooth slows the WiFi
            self.car_text.set("BLE disabled (--no-ble)")
        else:
            threading.Thread(target=self._ble_thread, daemon=True).start()

        root.bind_all("<KeyPress>", self._on_key_press)
        root.bind_all("<KeyRelease>", self._on_key_release)
        root.protocol("WM_DELETE_WINDOW", self._close)
        self._pump()

    # ---- UI building -------------------------------------------------
    def _card(self, parent, title):
        outer = tk.Frame(parent, bg=CARD, highlightbackground=LINE, highlightthickness=1)
        tk.Label(outer, text=title.upper(), bg=CARD, fg=MUTED, font=(FONT, 9, "bold")).pack(anchor="w", padx=14, pady=(10, 0))
        body = tk.Frame(outer, bg=CARD)
        body.pack(fill="both", expand=True, padx=14, pady=(4, 12))
        return outer, body

    def _badge(self, parent, label, var):
        f = tk.Frame(parent, bg=CARD, highlightbackground=LINE, highlightthickness=1)
        dot = tk.Canvas(f, width=12, height=12, bg=CARD, highlightthickness=0)
        dot.pack(side="left", padx=(10, 4), pady=8)
        oval = dot.create_oval(2, 2, 10, 10, fill=MUTED, outline="")
        tk.Label(f, text=label, bg=CARD, fg=MUTED, font=(FONT, 9, "bold")).pack(side="left")
        tk.Label(f, textvariable=var, bg=CARD, fg=TEXT, font=(FONT, 10)).pack(side="left", padx=(6, 12))
        f.dot, f.oval = dot, oval
        return f

    def _build_ui(self):
        root = self.root
        root.columnconfigure(0, weight=3)
        root.columnconfigure(1, weight=2, minsize=430)
        root.rowconfigure(1, weight=1)

        # header
        header = tk.Frame(root, bg=BG)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=16, pady=(14, 8))
        tk.Label(header, text="LitterVision", bg=BG, fg=TEXT, font=(FONT, 20, "bold")).pack(side="left")
        tk.Label(header, text="  car dashboard", bg=BG, fg=MUTED, font=(FONT, 12)).pack(side="left", pady=(7, 0))
        self.car_badge = self._badge(header, "CAR (BLE)", self.car_text)
        self.car_badge.pack(side="right")
        self.cam_badge = self._badge(header, "CAMERA (WiFi)", self.cam_text)
        self.cam_badge.pack(side="right", padx=(0, 8))
        self._button(header, "Path map", self.open_path_window).pack(side="right", padx=(0, 8))

        # camera column
        left = tk.Frame(root, bg=BG)
        left.grid(row=1, column=0, sticky="nsew", padx=(16, 8))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        cam_card, cam_body = self._card(left, "Camera")
        cam_card.grid(row=0, column=0, sticky="nsew")
        self.video = tk.Label(cam_body, bg="#000000", fg=MUTED, font=(FONT, 12), text="No camera signal")
        self.video.pack(fill="both", expand=True)
        self.video.bind("<Configure>", lambda e: setattr(self, "last_frame_id", -1))
        bar = tk.Frame(cam_body, bg=CARD)
        bar.pack(fill="x", pady=(8, 0))
        self._button(bar, "Save frame", self._save_frame).pack(side="left")
        tk.Label(bar, text="Rotate", bg=CARD, fg=MUTED, font=(FONT, 9)).pack(side="left", padx=(16, 4))
        combo = ttk.Combobox(bar, textvariable=self.rotate_var, values=list(ROTATIONS), width=6, state="readonly")
        combo.pack(side="left")
        combo.bind("<<ComboboxSelected>>", lambda e: setattr(self.camera, "rotate", ROTATIONS[self.rotate_var.get()]))
        self.cam_info = tk.Label(bar, text="", bg=CARD, fg=MUTED, font=(FONT, 9))
        self.cam_info.pack(side="right")
        tk.Label(bar, textvariable=self.run_status_var, bg=CARD, fg=WARN, font=(FONT, 10, "bold")).pack(side="left", padx=(16, 0))

        # right column
        right = tk.Frame(root, bg=BG)
        right.grid(row=1, column=1, sticky="nsew", padx=(8, 16))
        right.columnconfigure(0, weight=1)

        # distance
        dist_card, d = self._card(right, "Distance (ultrasonic)")
        dist_card.grid(row=0, column=0, sticky="ew")
        row = tk.Frame(d, bg=CARD)
        row.pack(fill="x")
        self.dist_label = tk.Label(row, textvariable=self.dist_var, bg=CARD, fg=TEXT, font=(FONT, 30, "bold"))
        self.dist_label.pack(side="left")
        tk.Label(row, text=" cm", bg=CARD, fg=MUTED, font=(FONT, 14)).pack(side="left", pady=(14, 0))
        tk.Label(row, textvariable=self.dist_note, bg=CARD, fg=MUTED, font=(FONT, 10)).pack(side="right")
        self.dist_bar = tk.Canvas(d, height=10, bg=CARD_HI, highlightthickness=0)
        self.dist_bar.pack(fill="x", pady=(6, 0))

        # motion
        imu_card, m = self._card(right, "Motion (MPU-6050)")
        imu_card.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self.dial = tk.Canvas(m, width=104, height=104, bg=CARD, highlightthickness=0)
        self.dial.pack(side="left", padx=(0, 14))
        info = tk.Frame(m, bg=CARD)
        info.pack(side="left", fill="x", expand=True)
        self.heading_label = tk.Label(info, textvariable=self.heading_var, bg=CARD, fg=TEXT, font=(FONT, 15, "bold"))
        self.heading_label.pack(anchor="w")
        for var in (self.spin_var, self.pitch_var, self.roll_var):
            tk.Label(info, textvariable=var, bg=CARD, fg=MUTED, font=(FONT, 10)).pack(anchor="w")
        btns = tk.Frame(info, bg=CARD)
        btns.pack(anchor="w", pady=(8, 0))
        self._button(btns, "Reset heading", lambda: self._send("Z")).pack(side="left")
        self._button(btns, "Calibrate (keep still)", lambda: self._send("C")).pack(side="left", padx=6)
        self._draw_dial()

        # drive
        drive_card, dr = self._card(right, "Drive  (arrows / WASD)")
        drive_card.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        for c in range(3):
            dr.columnconfigure(c, weight=1)
        # corner buttons: turn in place by an exact angle with the gyro (to test how accurate it is)
        for label, delta, r, c in (("↶ 90°", 90, 0, 0), ("↷ -90°", -90, 0, 2), ("↶ 180°", 180, 2, 0)):
            self._button(dr, label, lambda dlt=delta: self.turn_by(dlt)).grid(row=r, column=c, padx=3, pady=3, sticky="nsew")
        pad = [("▲", "U", 0, 1), ("◀", "L", 1, 0), ("STOP", "S", 1, 1), ("▶", "R", 1, 2), ("▼", "D", 2, 1)]
        for label, cmd, r, c in pad:
            b = self._button(dr, label, None, big=True, danger=(cmd == "S"))
            b.grid(row=r, column=c, padx=3, pady=3, sticky="nsew")
            if cmd == "S":
                b.configure(command=lambda: self._send("S"))
            else:
                b.bind("<ButtonPress-1>", lambda e, k=cmd: self._drive_button(k))
                b.bind("<ButtonRelease-1>", lambda e: self._drive_button_release())
                b.bind("<Leave>", lambda e: self._drive_button_release())  # only stops if the button was held
        # one speed slider for all wheels (sent as P<percent>)
        self.speed_dragging, self.last_speed_send, self.pending_speed = False, 0.0, None
        speed_row = tk.Frame(dr, bg=CARD)
        speed_row.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        tk.Label(speed_row, text="Speed", bg=CARD, fg=MUTED, font=(FONT, 9)).pack(side="left")
        self.car_speed_var = tk.StringVar(value=f"{CAR_SPEED_DEFAULT}%")
        tk.Label(speed_row, textvariable=self.car_speed_var, bg=CARD, fg=ACCENT, font=(FONT, 10, "bold")).pack(side="right")
        self.car_speed = self._scale(speed_row, CAR_SPEED_MIN, 100, CAR_SPEED_DEFAULT, self._speed_changed)
        self.car_speed.pack(side="left", fill="x", expand=True, padx=8)
        self.car_speed.bind("<ButtonPress-1>", lambda e: setattr(self, "speed_dragging", True))
        self.car_speed.bind("<ButtonRelease-1>", lambda e: self._speed_released())

        # servos (full width)
        servo_card, s = self._card(root, "Robotic arm servos")
        servo_card.grid(row=2, column=0, columnspan=2, sticky="ew", padx=16, pady=(10, 14))
        self.servo_scales, self.speed_scales, self.servo_values = [], [], []
        self.dragging, self.last_servo_send, self.pending_servo_send = set(), {}, {}
        for i, name in enumerate(SERVO_NAMES):
            s.columnconfigure(i, weight=1, uniform="servo")
            col = tk.Frame(s, bg=CARD)
            col.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 10, 0))
            top = tk.Frame(col, bg=CARD)
            top.pack(fill="x")
            tk.Label(top, text=f"{i} · {name}", bg=CARD, fg=TEXT, font=(FONT, 10, "bold")).pack(side="left")
            val = tk.StringVar(value="90°")
            self.servo_values.append(val)
            tk.Label(top, textvariable=val, bg=CARD, fg=ACCENT, font=(FONT, 10, "bold")).pack(side="right")
            scale = self._scale(col, 0, 180, 90, lambda v, idx=i: self._slider_changed(idx, v))
            scale.pack(fill="x")
            scale.bind("<ButtonPress-1>", lambda e, idx=i: self.dragging.add(idx))
            scale.bind("<ButtonRelease-1>", lambda e, idx=i: self._servo_released(idx))
            self.servo_scales.append(scale)
            sp = tk.Frame(col, bg=CARD)
            sp.pack(fill="x")
            sp_var = tk.StringVar(value=f"{SERVO_SPEED}°/s")
            tk.Label(sp, text="speed", bg=CARD, fg=MUTED, font=(FONT, 8)).pack(side="left")
            tk.Label(sp, textvariable=sp_var, bg=CARD, fg=MUTED, font=(FONT, 8)).pack(side="right")
            speed = self._scale(sp, 1, 180, SERVO_SPEED, lambda v, var=sp_var: var.set(f"{int(float(v))}°/s"), thin=True)
            speed.pack(side="left", fill="x", expand=True, padx=6)
            self.speed_scales.append(speed)
        presets = tk.Frame(s, bg=CARD)
        presets.grid(row=1, column=0, columnspan=len(SERVO_NAMES), sticky="w", pady=(8, 0))
        self._button(presets, "All center 90°", lambda: self._set_all_servos(90)).pack(side="left")
        self._button(presets, "All 0°", lambda: self._set_all_servos(0)).pack(side="left", padx=6)
        self._button(presets, "All 180°", lambda: self._set_all_servos(180)).pack(side="left")

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TCombobox", fieldbackground=CARD_HI, background=CARD_HI, foreground=TEXT, arrowcolor=TEXT)

    def _button(self, parent, text, command, big=False, danger=False):
        bg = BAD if danger else CARD_HI
        b = tk.Button(
            parent, text=text, command=command, bg=bg, fg=TEXT, activebackground=ACCENT, activeforeground="#ffffff",
            relief="flat", bd=0, highlightthickness=0, takefocus=0, cursor="hand2",
            font=(FONT, 13 if big else 9, "bold" if big else "normal"), padx=10, pady=5 if big else 4,
        )
        return b

    def _scale(self, parent, lo, hi, init, command, thin=False):
        sc = tk.Scale(
            parent, from_=lo, to=hi, orient="horizontal", resolution=1, showvalue=False, command=command,
            bg=ACCENT, fg=TEXT, troughcolor=CARD_HI, activebackground="#58a6ff", highlightthickness=0, bd=0,
            sliderrelief="flat", sliderlength=16, width=8 if thin else 14, takefocus=0,
        )
        sc.set(init)
        return sc

    # ---- thread-safe UI updates ---------------------------------------
    def _post(self, fn):
        self.ui_queue.put(fn)

    def _pump(self):
        try:
            while True:
                self.ui_queue.get_nowait()()
        except queue.Empty:
            pass
        self._refresh_camera()
        self._refresh_badges()
        self._path_tick()
        if not self.closing:
            self.root.after(30, self._pump)

    def _set_dot(self, badge, color):
        badge.dot.itemconfigure(badge.oval, fill=color)

    def _refresh_badges(self):
        cam = self.camera
        if cam.connected and cam.message == "Live":
            self.cam_text.set(f"Live · {cam.fps:.0f} FPS")
            self._set_dot(self.cam_badge, GOOD)
        else:
            self.cam_text.set(cam.message)
            self._set_dot(self.cam_badge, WARN if cam.connected else BAD)
        color = GOOD if self.connected.is_set() else (WARN if "Searching" in self.car_text.get() or "Connecting" in self.car_text.get() else BAD)
        self._set_dot(self.car_badge, color)

    def _refresh_camera(self):
        fid, rgb = self.camera.snapshot()
        if rgb is None:
            if not self.camera.connected:
                self.video.configure(image="", text="No camera signal")
                self.photo = None
            return
        if fid == self.last_frame_id:
            return
        self.last_frame_id = fid
        w, h = self.video.winfo_width(), self.video.winfo_height()
        if w < 20 or h < 20:
            return
        ih, iw = rgb.shape[:2]
        scale = min(w / iw, h / ih)
        size = (max(1, int(iw * scale)), max(1, int(ih * scale)))
        img = Image.fromarray(rgb).resize(size, Image.BILINEAR)
        self.photo = ImageTk.PhotoImage(img)
        self.video.configure(image=self.photo, text="")
        with self.camera.lock:
            kb = len(self.camera.last_jpeg or b"") / 1000
        # Link speed = frame size x FPS: big (detailed / close-up) frames mean few FPS on the same link.
        self.cam_info.configure(text=f"{iw}×{ih}  ·  {kb:.0f} KB/frame  ·  link {self.camera.link_kbps:.0f} KB/s")

    def _save_frame(self):
        with self.camera.lock:
            jpeg = self.camera.last_jpeg
        if jpeg is None:
            return
        CAPTURE_DIR.mkdir(exist_ok=True)
        path = CAPTURE_DIR / f"frame_{time.strftime('%Y%m%d_%H%M%S')}_{self.saved:03d}.jpg"
        path.write_bytes(jpeg)  # the camera's exact JPEG, unrotated: ideal for YOLO training images
        self.saved += 1
        self.cam_info.configure(text=f"saved {path.name}")

    # ---- sensors ------------------------------------------------------
    def _draw_dial(self):
        c = self.dial
        c.delete("all")
        cx, cy, r = 52, 52, 44
        c.create_oval(cx - r, cy - r, cx + r, cy + r, outline=LINE, width=2)
        c.create_text(cx, cy - r + 8, text="0°", fill=MUTED, font=(FONT, 8))
        h = math.radians(self.heading_deg)
        tip = (cx - (r - 10) * math.sin(h), cy - (r - 10) * math.cos(h))  # + heading = counter-clockwise (left)
        c.create_line(cx, cy, *tip, width=4, arrow="last", arrowshape=(11, 13, 5), fill=ACCENT)
        c.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill=ACCENT, outline="")

    def _on_distance(self, _sender, data):
        try:
            cm = float(bytes(data).decode("ascii"))
        except ValueError:
            return
        self._post(lambda: self._show_distance(cm))

    def _show_distance(self, cm):
        bar = self.dist_bar
        bar.delete("all")
        w = max(bar.winfo_width(), 1)
        self.dist_cm = None if cm < 0 else cm
        if cm < 0:
            self.dist_var.set("--")
            self.dist_note.set("nothing in range")
            self.dist_label.configure(fg=MUTED)
            return
        color = BAD if cm < DIST_STOP_CM else WARN if cm < DIST_WARN_CM else GOOD
        self.dist_var.set(f"{cm:.0f}")
        self.dist_note.set("OBSTACLE CLOSE" if cm < DIST_STOP_CM else "careful" if cm < DIST_WARN_CM else "clear")
        self.dist_label.configure(fg=color)
        fill = w * (1 - min(cm, 150) / 150)  # fuller bar = closer
        bar.create_rectangle(0, 0, max(fill, 3), 10, fill=color, outline="")

    def _on_mpu(self, _sender, data):
        text = bytes(data).decode("ascii", errors="ignore").strip()
        if text == "NA":
            self._post(lambda: self.heading_var.set("MPU-6050 not found"))
            return
        try:
            yaw, pitch, roll, gx, gy, gz, ax, ay, az = (float(v) for v in text.split(","))
        except ValueError:
            return
        stamp = time.monotonic()       # when the reading arrived (the UI thread handles it a moment later)
        self._post(lambda: self._show_mpu(yaw * YAW_SIGN, pitch * PITCH_SIGN, roll * ROLL_SIGN, gz * YAW_SIGN, stamp))

    @staticmethod
    def _describe(value, positive, negative):
        return "level" if abs(value) < 1 else f"{abs(value):.0f}° {positive if value > 0 else negative}"

    def _show_mpu(self, yaw, pitch, roll, gz, stamp=None):
        self.tracker.feed_yaw(yaw, resync=self._yaw_resync, gz=gz, now=stamp)
        self._yaw_resync = False
        self.gz = gz
        self.heading_deg = yaw
        if abs(yaw) < 1:
            self.heading_var.set("Straight (0°)")
        else:
            self.heading_var.set(f"Turned {abs(yaw):.0f}° {'left' if yaw > 0 else 'right'}")
        spin = "stopped" if abs(gz) < 2 else f"{abs(gz):.0f}°/s {'left' if gz > 0 else 'right'}"
        self.spin_var.set(f"Turning speed: {spin}")
        self.pitch_var.set(f"Front-back tilt: {self._describe(pitch, 'front up', 'front down')}")
        self.roll_var.set(f"Side tilt: {self._describe(roll, 'leaning right', 'leaning left')}")
        self._draw_dial()

    def _reset_readouts(self):
        self.dist_var.set("--")
        self.dist_note.set("waiting for sensor")
        self.dist_label.configure(fg=MUTED)
        self.dist_bar.delete("all")
        self.heading_var.set("--")
        self.spin_var.set("Turning speed: --")
        self.pitch_var.set("Front-back tilt: --")
        self.roll_var.set("Side tilt: --")
        self.heading_deg = 0.0
        self.dist_cm = None
        self._yaw_resync = True        # the car may have rebooted: its yaw starts from 0 again
        self._draw_dial()

    # ---- BLE ----------------------------------------------------------
    def _ble_thread(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._ble_main())
        except Exception as exc:
            self._post(lambda e=str(exc): self.car_text.set(f"BLE error: {e}"))
        finally:
            pending = asyncio.all_tasks(self.loop)
            for task in pending:
                task.cancel()
            if pending:
                self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self.loop.close()

    async def _ble_main(self):
        while not self.stop_event.is_set():
            self._post(lambda: self.car_text.set("Searching for AI_Car..."))
            try:
                device = await BleakScanner.find_device_by_name(CAR_NAME, timeout=8)
                if device is None:
                    self._post(lambda: self.car_text.set("AI_Car not found - retrying"))
                    continue
                self._post(lambda: self.car_text.set("Connecting..."))
                client = BleakClient(device)
                await client.connect()
                self.client = client
                self.characteristic = None
                for service in client.services:
                    for char in service.characteristics:
                        if char.uuid.lower() == CHAR_UUID.lower():
                            self.characteristic = char
                if self.characteristic is None:
                    self._post(lambda: self.car_text.set("Control characteristic missing"))
                    await client.disconnect()
                    await asyncio.sleep(2)
                    continue
                self.write_response = "write" in self.characteristic.properties
                for uuid, handler in ((DISTANCE_CHAR_UUID, self._on_distance), (MPU_CHAR_UUID, self._on_mpu)):
                    try:
                        await client.start_notify(uuid, handler)
                    except Exception:
                        pass
                self.connected.set()
                self._post(self._send_speed)   # make the car match the slider after (re)connecting
                self._post(lambda: self.car_text.set("Connected to AI_Car"))
                while not self.stop_event.is_set() and client.is_connected:
                    await asyncio.sleep(0.1)
                if client.is_connected:
                    await client.write_gatt_char(self.characteristic, b"S", response=self.write_response)
                    await client.disconnect()
            except Exception as exc:
                self._post(lambda e=str(exc): self.car_text.set(f"BLE: {e[:40]}"))
            finally:
                self.connected.clear()
                self.client = None
                self._post(self._reset_readouts)
            if not self.stop_event.is_set():
                await asyncio.sleep(1.5)

    async def _write(self, message):
        client, char = self.client, self.characteristic
        if client and client.is_connected and char:
            await client.write_gatt_char(char, message.encode("ascii"), response=self.write_response)

    def _send(self, message):
        if not self.connected.is_set() or not self.loop or not self.loop.is_running():
            return
        if message in ("U", "D", "L", "R", "S"):
            if self.runner.active and not self._runner_sending:
                self.runner.abort("manual control")       # you touched the controls: take over
            self.tracker.set_command(message)
        elif message in ("Z", "C"):
            self._yaw_resync = True                        # the car zeroes its heading on these
        asyncio.run_coroutine_threadsafe(self._write(message), self.loop)

    def _runner_send(self, message):
        self._runner_sending = True
        try:
            self._send(message)
        finally:
            self._runner_sending = False

    def _drive_button(self, cmd):
        self._pressed_button = True
        self._send(cmd)

    def _drive_button_release(self):
        if self._pressed_button:
            self._pressed_button = False
            self._send("S")

    # ---- path map: dead reckoning, plan, return -----------------------
    def open_path_window(self):
        if self.path_window is None:
            self.path_window = PathWindow(self)
        else:
            self.path_window.win.lift()

    def _path_ready(self):
        if not self.connected.is_set():
            self.runner.status = "Car not connected"
            return False
        return True

    def _path_tick(self):
        now = time.monotonic()
        dt = min(now - self._last_tick, 0.2)
        self._last_tick = now
        if not self.connected.is_set():
            if self.runner.active:
                self.runner.abort("car disconnected")
            self.tracker.set_command("S")
            return
        self.tracker.update(dt, int(self.car_speed.get()), now)
        self.run_status_var.set("" if self.runner.status == "Idle" else self.runner.status)
        if self.runner.active:
            self.runner.tick(now, self.gz)
        if self.path_window is not None:
            self.path_window.refresh()

    def path_return(self):
        """Drive the recorded path backwards: undo every move and turn, newest first."""
        if not self._path_ready() or self.runner.active:
            return
        self._send("S")
        steps = return_steps(self.tracker.corners)
        if not steps:
            self.runner.status = "Nothing recorded to return along"
            return
        self.tracker.recording, self.tracker.phase = False, "back"
        self.runner.start(steps, "Return", time.monotonic())

    def turn_by(self, delta):
        """Turn in place by `delta` degrees (+ = left) using the gyro, then report how close it got."""
        if not self._path_ready() or self.runner.active:
            return
        self._send("S")
        self.runner.start([["turn_to", self.tracker.heading + delta]], f"Turn {delta:+d}°", time.monotonic())

    def path_run_plan(self, waypoints):
        """Drive from here through the waypoints clicked on the map (recorded, so it can be returned along)."""
        if not self._path_ready() or self.runner.active:
            return
        self._send("S")
        steps = plan_steps(waypoints)
        if not steps:
            self.runner.status = "Click waypoints on the map first"
            return
        self.tracker.recording, self.tracker.phase = True, "out"
        self.runner.start(steps, "Run path", time.monotonic())

    def path_abort(self):
        self.runner.abort("aborted by you")
        self._send("S")

    def path_clear(self):
        self.runner.abort("path cleared")
        self.tracker.recording = True
        self.tracker.reset()
        if self.path_window is not None:
            self.path_window.refresh(force=True)

    def _runner_finished(self, ok):
        self.tracker.recording, self.tracker.phase = True, "out"
        if ok and self.runner.label == "Return":
            self.tracker.corners = [(self.tracker.x, self.tracker.y)]   # back home: nothing left to undo

    def path_test_drive(self):
        """Drive straight for TEST_DRIVE_S, then ask how far the car really went to calibrate cm/s."""
        if not self._path_ready() or self.runner.active:
            return
        speed = int(self.car_speed.get())
        self.runner.status = f"Test drive: {TEST_DRIVE_S:.0f} s at {speed}% ... measure how far it goes"
        self._send("U")
        self.root.after(int(TEST_DRIVE_S * 1000), lambda: self._finish_test_drive(speed))

    def _finish_test_drive(self, speed):
        self._send("S")
        parent = self.path_window.win if self.path_window else self.root
        measured = simpledialog.askfloat("Calibrate", "How far did the car really move (cm)?", parent=parent,
                                         minvalue=1, maxvalue=2000)
        if measured:
            self.tracker.cm_per_s_100 = measured / TEST_DRIVE_S / (speed / 100.0)
            if self.path_window:
                self.path_window.speed_var.set(f"{self.tracker.cm_per_s_100:.1f}")
            self.runner.status = f"Calibrated: {self.tracker.cm_per_s_100:.1f} cm/s at 100%"
        else:
            self.runner.status = "Calibration cancelled"
        self.path_clear()                  # the test drive isn't part of any real path

    # ---- wheel speed (one slider for all wheels) ----------------------
    def _speed_changed(self, value):
        self.car_speed_var.set(f"{int(float(value))}%")
        if self.speed_dragging:
            self._send_speed_throttled()

    def _send_speed(self):
        self._send(f"P{int(self.car_speed.get())}")

    def _send_speed_throttled(self):
        if self.pending_speed is not None:
            return
        wait = SERVO_SEND_INTERVAL - (time.monotonic() - self.last_speed_send)
        if wait <= 0:
            self.last_speed_send = time.monotonic()
            self._send_speed()
        else:
            self.pending_speed = self.root.after(int(wait * 1000) + 1, self._flush_speed)

    def _flush_speed(self):
        self.pending_speed = None
        self.last_speed_send = time.monotonic()
        self._send_speed()

    def _speed_released(self):
        self.speed_dragging = False
        if self.pending_speed is not None:
            self.root.after_cancel(self.pending_speed)
            self.pending_speed = None
        self._send_speed()

    # ---- servos -------------------------------------------------------
    def _slider_changed(self, index, value):
        self.servo_values[index].set(f"{int(float(value))}°")
        if index in self.dragging:
            self._send_servo_throttled(index)

    def _send_servo_throttled(self, index):
        if index in self.pending_servo_send:
            return
        wait = SERVO_SEND_INTERVAL - (time.monotonic() - self.last_servo_send.get(index, 0))
        if wait <= 0:
            self.last_servo_send[index] = time.monotonic()
            self._send_servo(index)
        else:
            self.pending_servo_send[index] = self.root.after(int(wait * 1000) + 1, lambda: self._flush_servo(index))

    def _flush_servo(self, index):
        self.pending_servo_send.pop(index, None)
        self.last_servo_send[index] = time.monotonic()
        self._send_servo(index)

    def _servo_released(self, index):
        self.dragging.discard(index)
        pending = self.pending_servo_send.pop(index, None)
        if pending is not None:
            self.root.after_cancel(pending)
        self._send_servo(index)

    def _send_servo(self, index):
        self._send(f"V{index} {int(self.servo_scales[index].get())} {int(self.speed_scales[index].get())}")

    def _set_all_servos(self, angle):
        for index, scale in enumerate(self.servo_scales):
            scale.set(angle)
            self._send_servo(index)

    # ---- keyboard / shutdown -----------------------------------------
    def _on_key_press(self, event):
        if event.keysym == "Escape":
            self._close()
            return
        key = event.keysym.lower()
        cmd = {"up": "U", "down": "D", "left": "L", "right": "R", "w": "U", "s": "D", "a": "L", "d": "R"}.get(key)
        if cmd and self.active_key != key:  # ignore auto-repeat while held
            self.active_key = key
            self._send(cmd)

    def _on_key_release(self, event):
        key = event.keysym.lower()
        if key == self.active_key:
            self.active_key = None
            self._send("S")

    def _close(self):
        if self.closing:
            return
        self.closing = True
        self._send("S")
        self.stop_event.set()
        self.camera.stop_event.set()
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def main():
    root = tk.Tk()
    Dashboard(root)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
