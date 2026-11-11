"""Plain-language messages for camera connection problems (used by camera_yolo.py and camera_serial.py).

Instead of a Python traceback, the viewers print what is most likely wrong: the PC is on the wrong WiFi, the ESP32 is
off or out of range, nothing is listening on the stream port, the COM port is busy, and so on.
"""
import subprocess

BRIDGE_SSID = "ESP32-BRIDGE"
BRIDGE_PASSWORD = "12345678"


def _current_wifi():
    """Name of the WiFi network this PC is connected to; '' if it is not connected; None if unknown."""
    try:
        out = subprocess.run(["netsh", "wlan", "show", "interfaces"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return None
    state, ssid = "", ""
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


def _answers_ping(host):
    """True if the host replies, False if it does not, None if ping could not be run."""
    try:
        out = subprocess.run(["ping", "-n", "1", "-w", "1500", host], capture_output=True, text=True, timeout=8).stdout
    except Exception:
        return None
    return "TTL=" in out


def tcp_connect_message(exc, host, port):
    """Explain why connecting to host:port failed."""
    wifi = _current_wifi()
    on_bridge_net = host.startswith("192.168.4.")
    lines = [f"Cannot connect to the camera at {host}:{port}."]

    refused = isinstance(exc, ConnectionRefusedError)  # the device itself replied "no": it is alive
    unreachable = getattr(exc, "winerror", None) in (10051, 10065) or getattr(exc, "errno", None) in (101, 113)

    if refused:
        lines += ["The device answers, but nothing is listening on the stream port.",
                  "-> The bridge program is not running (it may be restarting or stuck in flash mode). "
                  "Press its EN/reset button and try again."]
    elif wifi == "" and on_bridge_net:
        lines += ["This PC is not connected to any WiFi network.",
                  f"-> Join the WiFi '{BRIDGE_SSID}' (password {BRIDGE_PASSWORD}), then run this again."]
    elif wifi and on_bridge_net and wifi != BRIDGE_SSID:
        lines += [f"This PC is connected to '{wifi}', not to the camera's network '{BRIDGE_SSID}'.",
                  f"-> Switch the WiFi to '{BRIDGE_SSID}' (password {BRIDGE_PASSWORD}), then run this again."]
    elif unreachable or _answers_ping(host) is False:
        lines += [f"The device does not answer (no ping reply from {host}).",
                  "-> The ESP32 is probably switched off, unplugged or out of range. Check its power, "
                  "wait about 10 seconds after switching it on, and make sure it is close to the laptop."]
    else:
        lines += ["The device answers, but the connection timed out.",
                  "-> Another viewer window is probably already using the stream (the bridge serves one "
                  "viewer at a time). Close other camera windows and try again."]
    lines.append(f"(technical detail: {type(exc).__name__}: {exc})")
    return "\n".join(lines)


def serial_open_message(exc, port):
    """Explain why a COM port could not be opened."""
    return "\n".join([
        f"Cannot open the serial port {port}.",
        "-> Is the ESP32-CAM / board plugged in with a data USB cable? Is another program using the port "
        "(the Arduino Serial Monitor, another viewer)? Close it and try again.",
        f"(technical detail: {type(exc).__name__}: {exc})",
    ])


def connection_lost_message(exc, via_wifi):
    """Explain why a stream that was working has stopped."""
    if via_wifi:
        return "\n".join([
            "The connection to the camera was lost.",
            "-> The ESP32 bridge was probably unplugged or restarted, or the WiFi dropped. Check its power, "
            f"make sure the PC is still on '{BRIDGE_SSID}', and run this again.",
            f"(technical detail: {type(exc).__name__}: {exc})",
        ])
    return "\n".join([
        "The USB serial port stopped responding.",
        "-> The board was probably unplugged or reset. Plug it back in and run this again.",
        f"(technical detail: {type(exc).__name__}: {exc})",
    ])


NO_FRAMES_MESSAGE = (
    "Connected, but no camera frames are arriving yet.\n"
    "-> Check that the ESP32-CAM is powered and that its U0T wire goes to pin G16 of the bridge "
    "(and that the grounds are joined). Still waiting..."
)
