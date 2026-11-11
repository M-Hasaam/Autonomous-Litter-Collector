# LitterVision

```
Mobile_app/            Android app (open this folder in Android Studio; gradlew lives here)
Embedded/
  esp32-cam/           ESP32-CAM firmware (CameraWebServer), WiFi test script (scripts/), captures/
  esp32/               ESP32 WROOM devkit firmware (WifiBridge) + its PC viewers (WifiBridge/scripts: camera_yolo, camera_distance, camera_serial)
  esp32-s3/            ESP32-S3 AI_Car (BLE car + 4 servos): AI_Car.ino and control_car.py
  arduino/             Arduino sketches and PC tools (AI_CAR, HC_SR04_Test, Robotic_ARM)
Training/
  litter_detector/     the app's model: train / export / predict
  paperball_roboflow/  Roboflow paper-ball dataset, repair script, trainer, trained model
  paperball_yolo_temp/ temporary paper-ball model built from camera captures
```

Datasets, runs and trained weights stay on disk and are git-ignored.
