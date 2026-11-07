import asyncio
import queue

from bleak import BleakClient, BleakScanner
from pynput import keyboard

DEVICE_NAME = "AI_Car"

CHAR_UUID = "87654321-4321-4321-4321-BA0987654321"

# Keyboard commands
KEY_MAP = {
    keyboard.Key.up: "U",
    keyboard.Key.down: "D",
    keyboard.Key.left: "L",
    keyboard.Key.right: "R",

    "w": "U",
    "s": "D",
    "a": "L",
    "d": "R",
}

command_queue = queue.Queue()
pressed_keys = set()
running = True


def on_press(key):
    global running

    if key == keyboard.Key.esc:
        running = False
        command_queue.put("S")
        return False

    command = KEY_MAP.get(key)

    if command is None and hasattr(key, "char") and key.char:
        command = KEY_MAP.get(key.char.lower())

    if command:
        pressed_keys.add(key)
        command_queue.put(command)


def on_release(key):
    command = KEY_MAP.get(key)

    if command is None and hasattr(key, "char") and key.char:
        command = KEY_MAP.get(key.char.lower())

    if command:
        pressed_keys.discard(key)
        command_queue.put("S")


async def main():
    global running

    print("Searching for AI_Car...")

    device = await BleakScanner.find_device_by_name(
        DEVICE_NAME,
        timeout=15
    )

    if device is None:
        print("AI_Car not found. Check that ESP32 is powered on.")
        return

    print(f"Found: {device.name}")
    print("Connecting...")

    async with BleakClient(device) as client:
        if not client.is_connected:
            print("Connection failed!")
            return

        print("Connected to AI_Car!")

        # Find the writable characteristic
        characteristic = None

        for service in client.services:
            for char in service.characteristics:
                if char.uuid.lower() == CHAR_UUID.lower():
                    characteristic = char
                    break

        if characteristic is None:
            print("Control characteristic not found!")
            return

        response = "write" in characteristic.properties

        print("\n===== ROBOT CONTROL =====")
        print("UP / W       = Forward")
        print("DOWN / S     = Reverse")
        print("LEFT / A     = Left")
        print("RIGHT / D    = Right")
        print("Release key  = Stop")
        print("ESC          = Exit")
        print("=========================\n")

        def keyboard_listener():
            with keyboard.Listener(
                on_press=on_press,
                on_release=on_release
            ) as listener:
                listener.join()

        import threading

        listener_thread = threading.Thread(
            target=keyboard_listener,
            daemon=True
        )
        listener_thread.start()

        last_command = None

        while running and client.is_connected:
            try:
                command = command_queue.get_nowait()

                if command != last_command:
                    await client.write_gatt_char(
                        characteristic,
                        command.encode("ascii"),
                        response=response
                    )

                    print(f"Sent: {command}")
                    last_command = command

            except queue.Empty:
                pass

            await asyncio.sleep(0.02)

        # Always stop before exiting
        if client.is_connected:
            await client.write_gatt_char(
                characteristic,
                b"S",
                response=response
            )

        print("Robot stopped. Disconnected.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")