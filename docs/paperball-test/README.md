# Paper-ball detector test page

A single static page (`index.html`) that runs a YOLO11n paper-ball model (640x640, 1 class, `models/*.onnx`) in the browser with
onnxruntime-web (WASM, loaded from a CDN). It opens the phone camera, draws the boxes live and shows the time per frame.
"Pick photo" tests one picture; the confidence slider shows weaker detections. No picture leaves the device.

* Needs HTTPS (or localhost) for the camera: iOS Safari refuses the camera on plain http.
* A **Model** switch picks between `v2.1` (default) and `v2.0`; the first load of each downloads about 10 MB, switching back is instant. `?model=v2.0` selects one, `?model=<url>` loads another ONNX file (same input/output shape); `?image=<url>` runs once on a picture.
* To add a model, put the `.onnx` export in `models/` and add it to the `MODELS` list at the top of the script in `index.html`. Ultralytics YOLO11 is AGPL-3.0.
* Speed: one thread of WASM, so a few frames per second on a phone. The Android app uses LiteRT and is faster.
