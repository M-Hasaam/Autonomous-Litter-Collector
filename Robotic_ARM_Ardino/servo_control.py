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
        self.root.geometry("620x720")
        self.root.configure(bg="#f4f5f7")

        self.arduino = None
        self.recorded_poses = []
        self.is_playing = False
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
        conn_frame.pack(fill="x", padx=15, pady=8)

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
        controls_frame.pack(fill="x", padx=15, pady=8)

        for i, cfg in enumerate(self.servo_config):
            row_frame = ttk.Frame(controls_frame, padding=5)
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

        # --- Global Presets Panel ---
        preset_frame = ttk.LabelFrame(self.root, text=" Quick Presets ", padding=10)
        preset_frame.pack(fill="x", padx=15, pady=8)

        ttk.Button(preset_frame, text="ALL 90° (Neutral)", command=lambda: self.set_all_angles(90)).pack(side="left", expand=True, fill="x", padx=3)
        ttk.Button(preset_frame, text="ALL 0° (Home)", command=lambda: self.set_all_angles(0)).pack(side="left", expand=True, fill="x", padx=3)
        ttk.Button(preset_frame, text="ALL 180° (Max)", command=lambda: self.set_all_angles(180)).pack(side="left", expand=True, fill="x", padx=3)
        ttk.Button(preset_frame, text="Toggle Claw (0°/90°)", command=self.toggle_claw).pack(side="left", expand=True, fill="x", padx=3)

        # --- Pose Recorder & Sequence Panel ---
        seq_frame = ttk.LabelFrame(self.root, text=" Pose Sequence Recorder ", padding=10)
        seq_frame.pack(fill="both", expand=True, padx=15, pady=8)

        btn_box = ttk.Frame(seq_frame)
        btn_box.pack(fill="x", pady=2)

        ttk.Button(btn_box, text="📸 Record Current Pose", command=self.record_pose).pack(side="left", padx=3)
        ttk.Button(btn_box, text="▶ Play Sequence", command=self.play_sequence_thread).pack(side="left", padx=3)
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
        self.angle_vars[idx].set(angle)
        self.send_command(idx, angle)

    def adjust_angle(self, idx, delta):
        curr = self.angle_vars[idx].get()
        new_angle = max(0, min(180, curr + delta))
        self.angle_vars[idx].set(new_angle)
        self.send_command(idx, new_angle)

    def set_all_angles(self, angle):
        for i in range(len(self.servo_config)):
            self.angle_vars[i].set(angle)
            self.send_command(i, angle)

    def toggle_claw(self):
        claw_idx = 3  # Index 3 is Claw
        curr = self.angle_vars[claw_idx].get()
        target = 0 if curr > 45 else 90
        self.angle_vars[claw_idx].set(target)
        self.send_command(claw_idx, target)

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
            for idx, angle in enumerate(pose):
                self.angle_vars[idx].set(angle)
                self.send_command(idx, angle)
                time.sleep(0.1)
            time.sleep(1.0) # Wait at each keyframe
        self.is_playing = False

if __name__ == "__main__":
    root = tk.Tk()
    app = ServoControllerApp(root)
    root.mainloop()