import cv2
import numpy as np
import requests
import threading
import time


class ESP32Camera:
    def __init__(self, url):
        self.url = url
        self.frame = None
        self.frame_number = 0
        self.frame_time = 0

        self.reconnects = 0
        self.running = False

        self.lock = threading.Lock()
        self.thread = None
        self.response = None

    def start(self):
        self.running = True

        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True
        )

        self.thread.start()

    def _capture_loop(self):
        session = requests.Session()

        while self.running:
            try:
                print("Connecting to ESP32-CAM...")

                self.response = session.get(
                    self.url,
                    stream=True,
                    timeout=(3, 3)
                )

                self.response.raise_for_status()
                print("Stream connected")

                buffer = bytearray()

                for chunk in self.response.iter_content(
                    chunk_size=4096
                ):
                    if not self.running:
                        break

                    if not chunk:
                        continue

                    buffer.extend(chunk)

                    # Prevent unlimited buffer growth
                    if len(buffer) > 1_000_000:
                        buffer = buffer[-200_000:]

                    while True:
                        start = buffer.find(b"\xff\xd8")
                        end = buffer.find(b"\xff\xd9", start + 2)

                        if start == -1 or end == -1:
                            break

                        jpeg = bytes(buffer[start:end + 2])
                        del buffer[:end + 2]

                        image_data = np.frombuffer(
                            jpeg,
                            dtype=np.uint8
                        )

                        new_frame = cv2.imdecode(
                            image_data,
                            cv2.IMREAD_COLOR
                        )

                        if new_frame is None:
                            continue

                        new_frame = cv2.rotate(
                            new_frame,
                            cv2.ROTATE_90_COUNTERCLOCKWISE
                        )

                        with self.lock:
                            self.frame = new_frame
                            self.frame_number += 1
                            self.frame_time = time.perf_counter()

            except requests.RequestException as error:
                if self.running:
                    self.reconnects += 1
                    print(f"Stream stalled: {error}")
                    print("Reconnecting...")
                    time.sleep(1)

            finally:
                if self.response is not None:
                    self.response.close()
                    self.response = None

        session.close()

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None, 0, 0

            return (
                self.frame.copy(),
                self.frame_number,
                self.frame_time
            )

    def stop(self):
        self.running = False

        if self.response is not None:
            self.response.close()

        if self.thread is not None:
            self.thread.join(timeout=4)


url = "http://192.168.137.30:81/stream"

camera = ESP32Camera(url)
camera.start()

start_time = time.perf_counter()
last_report_time = start_time
last_frame_count = 0
last_shown_number = -1

current_fps = 0

try:
    while True:
        frame, frame_number, frame_time = camera.get_frame()
        current_time = time.perf_counter()

        if frame is not None:
            frame_age_ms = (
                current_time - frame_time
            ) * 1000

            if frame_number != last_shown_number:
                last_shown_number = frame_number

                cv2.putText(
                    frame,
                    f"FPS: {current_fps:.1f}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Read age: {frame_age_ms:.0f} ms",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                cv2.imshow("ESP32-CAM", frame)

        report_duration = current_time - last_report_time

        if report_duration >= 1:
            new_frames = frame_number - last_frame_count
            current_fps = new_frames / report_duration

            total_duration = current_time - start_time
            average_fps = frame_number / total_duration

            if frame_time:
                seconds_without_frame = current_time - frame_time
            else:
                seconds_without_frame = 0

            print(
                f"FPS: {current_fps:.2f} | "
                f"Average: {average_fps:.2f} | "
                f"No-frame time: {seconds_without_frame:.2f}s | "
                f"Frames: {frame_number} | "
                f"Reconnects: {camera.reconnects}"
            )

            last_frame_count = frame_number
            last_report_time = current_time

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(0.001)

except KeyboardInterrupt:
    print("\nStopped using Ctrl+C")

finally:
    camera.stop()
    cv2.destroyAllWindows()

    runtime = time.perf_counter() - start_time
    average_fps = camera.frame_number / runtime

    print("\nFinal statistics")
    print(f"Runtime: {runtime:.1f} seconds")
    print(f"Frames: {camera.frame_number}")
    print(f"Average FPS: {average_fps:.2f}")
    print(f"Reconnects: {camera.reconnects}")