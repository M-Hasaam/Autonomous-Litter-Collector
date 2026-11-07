import asyncio
import threading
import tkinter as tk
from tkinter import ttk

from bleak import BleakClient, BleakScanner

DEVICE_NAME = "AI_Car"
CHAR_UUID = "87654321-4321-4321-4321-BA0987654321"

SERVICE_UUID = "12345678-1234-1234-1234-1234567890AB"

# Servo protocol: V<index> <angle> <speed deg/sec>, e.g. V0 90 60
SERVO_SPEED = 60
SERVO_NAMES = ["Base Rotation", "Shoulder", "Elbow", "Gripper"]

class AICarController:
    def __init__(self, root):
        self.root = root
        self.root.title("AI_Car BLE Controller")
        self.root.geometry("680x690")
        self.root.minsize(600, 620)

        self.loop = None
        self.client = None
        self.characteristic = None
        self.write_response = False
        self.connected = threading.Event()
        self.stop_event = threading.Event()
        self.loop_ready = threading.Event()
        self.pressed = set()
        self.last_direction = None
        self.listener = None
        self.closing = False

        self.status_var = tk.StringVar(value="Disconnected")
        self._build_ui()

        self.worker = threading.Thread(target=self._ble_thread, daemon=True)
        self.worker.start()
        self.root.bind_all("<KeyPress>", self._on_key_press)
        self.root.bind_all("<KeyRelease>", self._on_key_release)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=14)
        main.pack(fill="both", expand=True)

        header = ttk.Frame(main)
        header.pack(fill="x")
        ttk.Label(header, text="AI_Car Controller", font=("Segoe UI", 17, "bold")).pack(side="left")
        ttk.Label(header, textvariable=self.status_var).pack(side="right")

        ttk.Label(
            main,
            text="Drive: Arrow keys or W A S D. Release the key to stop.",
        ).pack(anchor="w", pady=(8, 10))

        drive = ttk.LabelFrame(main, text="Wheel control", padding=12)
        drive.pack(fill="x")

        for col in range(3):
            drive.columnconfigure(col, weight=1)

        self._drive_button(drive, "Forward  ↑", "U", 0, 1)
        self._drive_button(drive, "Left  ←", "L", 1, 0)
        self._drive_button(drive, "STOP", "S", 1, 1, stop_button=True)
        self._drive_button(drive, "Right  →", "R", 1, 2)
        self._drive_button(drive, "Reverse  ↓", "D", 2, 1)

        servos_frame = ttk.LabelFrame(main, text="Servo control", padding=12)
        servos_frame.pack(fill="both", expand=True, pady=(14, 0))

        self.servo_scales = []
        self.servo_values = []
        self.speed_scales = []

        for i, name in enumerate(SERVO_NAMES):
            row = ttk.Frame(servos_frame)
            row.pack(fill="x", pady=7)

            top = ttk.Frame(row)
            top.pack(fill="x")
            ttk.Label(top, text=f"Servo {i} — {name}", font=("Segoe UI", 10, "bold")).pack(side="left")

            value_var = tk.StringVar(value="90°")
            self.servo_values.append(value_var)
            ttk.Label(top, textvariable=value_var, width=6, anchor="e").pack(side="right")

            scale = tk.Scale(
                row,
                from_=0,
                to=180,
                orient="horizontal",
                resolution=1,
                showvalue=False,
                command=lambda value, index=i: self._slider_changed(index, value),
            )
            scale.set(90)
            scale.pack(fill="x")
            scale.bind("<ButtonRelease-1>", lambda event, index=i: self._send_servo(index))
            self.servo_scales.append(scale)

            speed_row = ttk.Frame(row)
            speed_row.pack(fill="x")
            ttk.Label(speed_row, text="Speed", foreground="gray").pack(side="left")
            speed_var = tk.StringVar(value=f"{SERVO_SPEED}°/s")
            ttk.Label(speed_row, textvariable=speed_var, width=8, anchor="e", foreground="gray").pack(side="right")
            speed_scale = tk.Scale(
                speed_row,
                from_=1,
                to=180,
                orient="horizontal",
                resolution=1,
                showvalue=False,
                length=200,
                command=lambda value, var=speed_var: var.set(f"{int(float(value))}°/s"),
            )
            speed_scale.set(SERVO_SPEED)
            speed_scale.pack(side="left", fill="x", expand=True, padx=8)
            self.speed_scales.append(speed_scale)

        preset_row = ttk.Frame(servos_frame)
        preset_row.pack(fill="x", pady=(8, 0))
        ttk.Button(preset_row, text="All center (90°)", command=lambda: self._set_all_servos(90)).pack(side="left")
        ttk.Button(preset_row, text="All 0°", command=lambda: self._set_all_servos(0)).pack(side="left", padx=6)
        ttk.Button(preset_row, text="All 180°", command=lambda: self._set_all_servos(180)).pack(side="left")

        footer = ttk.Label(
            main,
            text="BLE: AI_Car  |  Servo command: V<n> <angle> <speed>  |  Press Esc to stop/exit",
            foreground="gray",
        )
        footer.pack(anchor="w", pady=(12, 0))

    def _drive_button(self, parent, label, command, row, column, stop_button=False):
        button = ttk.Button(parent, text=label)
        button.grid(row=row, column=column, padx=5, pady=5, sticky="nsew")
        if stop_button:
            button.configure(command=lambda: self._send("S"))
        else:
            button.bind("<ButtonPress-1>", lambda event, cmd=command: self._start_drive(cmd))
            button.bind("<ButtonRelease-1>", lambda event: self._stop_drive())
            button.bind("<Leave>", lambda event: self._stop_drive())
        return button

    def _ble_thread(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop_ready.set()
        try:
            self.loop.run_until_complete(self._ble_main())
        except Exception as exc:
            self.root.after(0, lambda error=str(exc): self.status_var.set(f"BLE error: {error}"))
        finally:
            pending = asyncio.all_tasks(self.loop)
            for task in pending:
                task.cancel()
            if pending:
                self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self.loop.close()

    async def _ble_main(self):
        self.root.after(0, lambda: self.status_var.set("Searching for AI_Car..."))
        device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=20)

        if device is None:
            self.root.after(0, lambda: self.status_var.set("AI_Car not found"))
            return

        self.client = BleakClient(device)
        await self.client.connect()

        if not self.client.is_connected:
            self.root.after(0, lambda: self.status_var.set("Connection failed"))
            return

        for service in self.client.services:
            for char in service.characteristics:
                if char.uuid.lower() == CHAR_UUID.lower():
                    self.characteristic = char
                    break

        if self.characteristic is None:
            self.root.after(0, lambda: self.status_var.set("Control characteristic missing"))
            await self.client.disconnect()
            return

        self.write_response = "write" in self.characteristic.properties
        self.connected.set()
        self.root.after(0, lambda: self.status_var.set("Connected to AI_Car"))

        while not self.stop_event.is_set() and self.client.is_connected:
            await asyncio.sleep(0.1)

        if self.client.is_connected:
            await self.client.write_gatt_char(
                self.characteristic, b"S", response=self.write_response
            )
            await self.client.disconnect()

        self.connected.clear()
        self.root.after(0, lambda: self.status_var.set("Disconnected"))

    async def _write(self, message):
        if self.client and self.client.is_connected and self.characteristic:
            await self.client.write_gatt_char(
                self.characteristic,
                message.encode("ascii"),
                response=self.write_response,
            )

    def _send(self, message):
        if not self.connected.is_set() or not self.loop or not self.loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(self._write(message), self.loop)

    def _start_drive(self, command):
        self.last_direction = command
        self._send(command)

    def _stop_drive(self):
        self.last_direction = None
        self._send("S")

    def _slider_changed(self, index, value):
        angle = int(float(value))
        self.servo_values[index].set(f"{angle}°")

    def _send_servo(self, index):
        angle = int(self.servo_scales[index].get())
        speed = int(self.speed_scales[index].get())
        self._send(f"V{index} {angle} {speed}")

    def _set_all_servos(self, angle):
        for index, scale in enumerate(self.servo_scales):
            scale.set(angle)
            self._send_servo(index)

    def _on_key_press(self, event):
        if event.keysym == "Escape":
            self._send("S")
            self._close()
            return

        key = event.keysym.lower()
        command = None
        if key in ("up", "down", "left", "right"):
            command = {"up": "U", "down": "D", "left": "L", "right": "R"}[key]
        elif key in ("w", "a", "s", "d"):
            command = {"w": "U", "a": "L", "s": "D", "d": "R"}[key]

        if command:
            self.pressed.add(key)
            self._send(command)

    def _on_key_release(self, event):
        key = event.keysym.lower()
        if key in ("up", "down", "left", "right", "w", "a", "s", "d"):
            self.pressed.discard(key)
            self._send("S")

    def _close(self):
        if self.closing:
            return
        self.closing = True
        self._send("S")
        self.stop_event.set()
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def main():
    root = tk.Tk()
    app = AICarController(root)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
