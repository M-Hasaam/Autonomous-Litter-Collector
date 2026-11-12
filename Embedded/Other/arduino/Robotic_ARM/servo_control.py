import tkinter as tk
from tkinter import ttk, messagebox
import serial
import serial.tools.list_ports
import time
import threading

class ServoControllerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Robotic Arm 4-Axis Controller")
        self.root.geometry("640x820")
        self.root.configure(bg="#f4f5f7")

        self.arduino = None
        self.recorded_poses = []
        self.is_playing = False
        self.is_moving = False
        self._last_send_time = [0.0] * 4  # Rate limit per servo

        # Servo definitions: Pin, Default Name, Default Angle
        self.servo_config = [
            {"pin": 6,  "name": "Base Rotation", "angle": 90},
            {"pin": 9,  "name": "Shoulder",      "angle": 90},
            {"pin": 10, "name": "Elbow",         "angle": 90},
            {"pin": 11, "name": "Claw / Gripper","angle": 90}
        ]

        self.sliders = []
        self.entries = []
        self.angle_vars = []

        self.setup_ui()
        self.refresh_ports()

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use("clam")

        # --- Connection Panel ---
        conn_frame = ttk.LabelFrame(self.root, text=" Serial Connection ", padding=10)
        conn_frame.pack(fill="x", padx=15, pady=6)

        ttk.Label(conn_frame, text="Port:").pack(side="left", padx=(0, 5))
        self.port_combobox = ttk.Combobox(conn_frame, width=15, state="readonly")
        self.port_combobox.pack(side="left", padx=5)

        ttk.Button(conn_frame, text="🔄 Refresh", command=self.refresh_ports).pack(side="left", padx=5)
        self.btn_connect = ttk.Button(conn_frame, text="Connect", command=self.toggle_connection)
        self.btn_connect.pack(side="left", padx=5)

        self.lbl_status = ttk.Label(conn_frame, text="Disconnected", foreground="red")
        self.lbl_status.pack(side="left", padx=10)

        # --- Servo Controls Panel ---
        controls_frame = ttk.LabelFrame(self.root, text=" Servo Controls ", padding=10)
        controls_frame.pack(fill="x", padx=15, pady=6)

        for i, cfg in enumerate(self.servo_config):
            row_frame = ttk.Frame(controls_frame, padding=4)
            row_frame.pack(fill="x", pady=2)

            # Label
            lbl = ttk.Label(row_frame, text=f"[{i}] {cfg['name']} (Pin {cfg['pin']}):", width=22, font=("Helvetica", 9, "bold"))
            lbl.grid(row=0, column=0, sticky="w")

            # Var & Entry
            var = tk.IntVar(value=cfg["angle"])
            self.angle_vars.append(var)

            # -5 Deg Button
            btn_minus = ttk.Button(row_frame, text="-5°", width=4, command=lambda idx=i: self.adjust_angle(idx, -5))
            btn_minus.grid(row=0, column=1, padx=2)

            # Slider
            slider = ttk.Scale(row_frame, from_=0, to=180, orient="horizontal", variable=var,
                               command=lambda val, idx=i: self.on_slider_move(idx, val))
            slider.grid(row=0, column=2, sticky="ew", padx=5)
            row_frame.columnconfigure(2, weight=1)
            self.sliders.append(slider)

            # +5 Deg Button
            btn_plus = ttk.Button(row_frame, text="+5°", width=4, command=lambda idx=i: self.adjust_angle(idx, 5))
            btn_plus.grid(row=0, column=3, padx=2)

            # Entry box
            entry = ttk.Entry(row_frame, textvariable=var, width=5)
            entry.grid(row=0, column=4, padx=5)
            entry.bind("<Return>", lambda event, idx=i: self.on_entry_submit(idx))
            self.entries.append(entry)

        # --- Motion Speed Panel ---
        speed_frame = ttk.LabelFrame(self.root, text=" Motion Speed Control ", padding=10)
        speed_frame.pack(fill="x", padx=15, pady=6)

        profile_row = ttk.Frame(speed_frame)
        profile_row.pack(fill="x", pady=2)

        ttk.Label(profile_row, text="Speed Profile:", font=("Helvetica", 9, "bold")).pack(side="left", padx=(0, 8))

        self.speed_mode_var = tk.StringVar(value="Medium")

        speed_presets = [
            ("🐢 Slow", "Slow", 2),
            ("🚶 Medium", "Medium", 5),
            ("⚡ Fast", "Fast", 8),
            ("🚀 Instant", "Instant", 0)
        ]

        for text, mode, spd in speed_presets:
            r = ttk.Radiobutton(
                profile_row,
                text=text,
                value=mode,
                variable=self.speed_mode_var,
                command=lambda s=spd, m=mode: self.set_speed_preset(m, s)
            )
            r.pack(side="left", padx=5)

        slider_row = ttk.Frame(speed_frame)
        slider_row.pack(fill="x", pady=(6, 2))

        ttk.Label(slider_row, text="Fine Tune (1-10):", width=14).pack(side="left", padx=(0, 5))
        self.speed_var = tk.IntVar(value=5)
        self.speed_slider = ttk.Scale(
            slider_row,
            from_=1,
            to=10,
            orient="horizontal",
            variable=self.speed_var,
            command=self.on_speed_slider_change
        )
        self.speed_slider.pack(side="left", fill="x", expand=True, padx=5)

        self.lbl_speed_val = ttk.Label(slider_row, text="Level 5", width=9)
        self.lbl_speed_val.pack(side="left", padx=5)

        # --- Global Presets Panel ---
        preset_frame = ttk.LabelFrame(self.root, text=" Quick Presets ", padding=10)
        preset_frame.pack(fill="x", padx=15, pady=6)

        ttk.Button(preset_frame, text="ALL 90° (Neutral)", command=lambda: self.set_all_angles_smooth(90)).pack(side="left", expand=True, fill="x", padx=3)
        ttk.Button(preset_frame, text="ALL 0° (Home)", command=lambda: self.set_all_angles_smooth(0)).pack(side="left", expand=True, fill="x", padx=3)
        ttk.Button(preset_frame, text="ALL 180° (Max)", command=lambda: self.set_all_angles_smooth(180)).pack(side="left", expand=True, fill="x", padx=3)
        ttk.Button(preset_frame, text="Toggle Claw (0°/90°)", command=self.toggle_claw_smooth).pack(side="left", expand=True, fill="x", padx=3)

        # --- Pose Recorder & Sequence Panel ---
        seq_frame = ttk.LabelFrame(self.root, text=" Pose Sequence Recorder ", padding=10)
        seq_frame.pack(fill="both", expand=True, padx=15, pady=6)

        btn_box = ttk.Frame(seq_frame)
        btn_box.pack(fill="x", pady=2)

        ttk.Button(btn_box, text="📸 Record Current Pose", command=self.record_pose).pack(side="left", padx=3)
        ttk.Button(btn_box, text="▶ Play Sequence", command=self.play_sequence_thread).pack(side="left", padx=3)
        ttk.Button(btn_box, text="⏹ Stop Motion", command=self.stop_motion).pack(side="left", padx=3)
        ttk.Button(btn_box, text="🗑 Clear Poses", command=self.clear_poses).pack(side="left", padx=3)

        self.pose_listbox = tk.Listbox(seq_frame, height=6, font=("Consolas", 9))
        self.pose_listbox.pack(fill="both", expand=True, pady=5)

    # --- Serial Communication Functions ---
    def refresh_ports(self):
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_combobox['values'] = ports
        if ports:
            self.port_combobox.current(0)

    def toggle_connection(self):
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
            self.arduino = None
            self.btn_connect.config(text="Connect")
            self.lbl_status.config(text="Disconnected", foreground="red")
        else:
            port = self.port_combobox.get()
            if not port:
                messagebox.showerror("Error", "No COM port selected!")
                return
            try:
                self.arduino = serial.Serial(port, 9600, timeout=1)
                time.sleep(2)
                self.btn_connect.config(text="Disconnect")
                self.lbl_status.config(text=f"Connected ({port})", foreground="green")
            except Exception as e:
                messagebox.showerror(
                    "Connection Error",
                    f"Could not open {port}:\n{e}\n\nTip: If Arduino IDE's Serial Monitor is open, CLOSE IT so Python can access the port."
                )

    def send_command(self, idx, angle):
        if self.arduino and self.arduino.is_open:
            cmd = f"{idx} {angle}\n"
            try:
                self.arduino.write(cmd.encode('utf-8'))
                self.arduino.flush()
            except Exception as e:
                print(f"Write error: {e}")
                self.lbl_status.config(text=f"Write Error: {e}", foreground="red")
        else:
            self.lbl_status.config(text="Not Connected! Select port & click Connect", foreground="orange")

    # --- Speed Handling ---
    def set_speed_preset(self, mode, val):
        self.speed_mode_var.set(mode)
        if val > 0:
            self.speed_var.set(val)
            self.lbl_speed_val.config(text=f"Level {val}")
        else:
            self.lbl_speed_val.config(text="Instant")

    def on_speed_slider_change(self, val):
        v = int(float(val))
        self.speed_var.set(v)
        self.lbl_speed_val.config(text=f"Level {v}")
        if self.speed_mode_var.get() == "Instant":
            self.speed_mode_var.set("Medium")

    def get_step_delay(self):
        if self.speed_mode_var.get() == "Instant":
            return 0.0
        lvl = max(1, min(10, self.speed_var.get()))
        # Speed 1: 0.040s (slow, ~25 deg/s)
        # Speed 5: 0.016s (~62 deg/s)
        # Speed 10: 0.004s (fast, ~250 deg/s)
        return 0.044 - (lvl * 0.004)

    def stop_motion(self):
        self.is_moving = False
        self.is_playing = False

    def move_to_pose(self, target_pose):
        self.is_moving = True
        start_pose = [var.get() for var in self.angle_vars]
        delay = self.get_step_delay()

        if delay <= 0:
            for idx, angle in enumerate(target_pose):
                self.angle_vars[idx].set(angle)
                self.send_command(idx, angle)
                time.sleep(0.01)
            self.is_moving = False
            return

        diffs = [abs(target_pose[i] - start_pose[i]) for i in range(len(target_pose))]
        max_diff = max(diffs)
        if max_diff == 0:
            self.is_moving = False
            return

        last_sent = list(start_pose)
        for step in range(1, max_diff + 1):
            if not self.is_moving:
                break
            frac = step / float(max_diff)
            for i in range(len(target_pose)):
                cur = int(round(start_pose[i] + (target_pose[i] - start_pose[i]) * frac))
                if cur != last_sent[i]:
                    self.angle_vars[i].set(cur)
                    self.send_command(i, cur)
                    last_sent[i] = cur
            time.sleep(delay)

        # Finalize target angles
        for i, target in enumerate(target_pose):
            self.angle_vars[i].set(target)
            self.send_command(i, target)

        self.is_moving = False

    def move_to_pose_thread(self, target_pose):
        if self.is_moving or self.is_playing:
            self.is_moving = False
            time.sleep(0.05)
        threading.Thread(target=self.move_to_pose, args=(target_pose,), daemon=True).start()

    # --- Motion Handling ---
    def on_slider_move(self, idx, val):
        angle = int(float(val))
        self.angle_vars[idx].set(angle)
        now = time.time()
        # Rate limit slider events to max 30Hz (every 30ms) to avoid buffer overflow
        if (now - self._last_send_time[idx]) >= 0.03:
            self._last_send_time[idx] = now
            self.send_command(idx, angle)

    def on_entry_submit(self, idx):
        angle = self.angle_vars[idx].get()
        angle = max(0, min(180, angle))
        delay = self.get_step_delay()
        if delay > 0:
            target_pose = [var.get() for var in self.angle_vars]
            target_pose[idx] = angle
            self.move_to_pose_thread(target_pose)
        else:
            self.angle_vars[idx].set(angle)
            self.send_command(idx, angle)

    def adjust_angle(self, idx, delta):
        curr = self.angle_vars[idx].get()
        new_angle = max(0, min(180, curr + delta))
        delay = self.get_step_delay()
        if delay > 0:
            target_pose = [var.get() for var in self.angle_vars]
            target_pose[idx] = new_angle
            self.move_to_pose_thread(target_pose)
        else:
            self.angle_vars[idx].set(new_angle)
            self.send_command(idx, new_angle)

    def set_all_angles_smooth(self, angle):
        self.move_to_pose_thread([angle] * len(self.servo_config))

    def toggle_claw_smooth(self):
        claw_idx = 3  # Index 3 is Claw
        curr = self.angle_vars[claw_idx].get()
        target = 0 if curr > 45 else 90
        target_pose = [var.get() for var in self.angle_vars]
        target_pose[claw_idx] = target
        self.move_to_pose_thread(target_pose)

    # --- Sequence Recording ---
    def record_pose(self):
        pose = [var.get() for var in self.angle_vars]
        self.recorded_poses.append(pose)
        pose_str = f"Pose {len(self.recorded_poses)}: Base={pose[0]}°, Shldr={pose[1]}°, Elbw={pose[2]}°, Claw={pose[3]}°"
        self.pose_listbox.insert(tk.END, pose_str)

    def clear_poses(self):
        self.recorded_poses.clear()
        self.pose_listbox.delete(0, tk.END)

    def play_sequence_thread(self):
        if not self.recorded_poses:
            messagebox.showinfo("Info", "No poses recorded!")
            return
        if self.is_playing:
            return
        threading.Thread(target=self.play_sequence, daemon=True).start()

    def play_sequence(self):
        self.is_playing = True
        for pose in self.recorded_poses:
            if not self.is_playing:
                break
            self.move_to_pose(pose)
            # Dwell at each keyframe
            start_dwell = time.time()
            while (time.time() - start_dwell) < 0.6 and self.is_playing:
                time.sleep(0.05)
        self.is_playing = False

if __name__ == "__main__":
    root = tk.Tk()
    app = ServoControllerApp(root)
    root.mainloop()
