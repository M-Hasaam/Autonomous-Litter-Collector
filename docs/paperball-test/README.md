# Paper-ball detector test page

A single static page (`index.html`) that runs `roboflow-paperball.onnx` (v2.0, YOLO11n, 640x640, 1 class) in the browser with
onnxruntime-web (WASM, loaded from a CDN). It opens the phone camera, draws the boxes live and shows the time per frame.
"Pick photo" tests one picture; the confidence slider shows weaker detections. No picture leaves the device.

* Needs HTTPS (or localhost) for the camera: iOS Safari refuses the camera on plain http.
* `?model=<url>` loads another ONNX file (same input/output shape as the model here); `?image=<url>` runs once on a picture.
* To test a new model, replace `roboflow-paperball.onnx` here with the new export (`train_roboflow.py` or the Colab notebook write it).
* The model file is a copy of `Training/paper_ball_detector/roboflow-paperball.onnx`; Ultralytics YOLO11 is AGPL-3.0.
* Speed: one thread of WASM, so a few frames per second on a phone. The Android app uses LiteRT and is faster.
