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
from tkinter import ttk

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
        self.photo = None
        self.last_frame_id = -1
        self.saved = 0

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
        pad = [("▲", "U", 0, 1), ("◀", "L", 1, 0), ("STOP", "S", 1, 1), ("▶", "R", 1, 2), ("▼", "D", 2, 1)]
        for label, cmd, r, c in pad:
            b = self._button(dr, label, None, big=True, danger=(cmd == "S"))
            b.grid(row=r, column=c, padx=3, pady=3, sticky="nsew")
            if cmd == "S":
                b.configure(command=lambda: self._send("S"))
            else:
                b.bind("<ButtonPress-1>", lambda e, k=cmd: self._send(k))
                b.bind("<ButtonRelease-1>", lambda e: self._send("S"))
                b.bind("<Leave>", lambda e: self._send("S"))
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
        self._post(lambda: self._show_mpu(yaw * YAW_SIGN, pitch * PITCH_SIGN, roll * ROLL_SIGN, gz * YAW_SIGN))

    @staticmethod
    def _describe(value, positive, negative):
        return "level" if abs(value) < 1 else f"{abs(value):.0f}° {positive if value > 0 else negative}"

    def _show_mpu(self, yaw, pitch, roll, gz):
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
        asyncio.run_coroutine_threadsafe(self._write(message), self.loop)

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
