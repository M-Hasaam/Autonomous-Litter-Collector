# Paper-ball detector test page

A single static page (`index.html`) that runs a YOLO11n paper-ball model (640x640, 1 class, `models/*.onnx`) in the browser with
onnxruntime-web (WASM, loaded from a CDN). It opens the phone camera, draws the boxes live and shows the time per frame.
"Pick photo" tests one picture; the confidence slider shows weaker detections. No picture leaves the device.

* Needs HTTPS (or localhost) for the camera: iOS Safari refuses the camera on plain http.
* A **Model** switch picks between `v2.1` (default) and `v2.0`; the first load of each downloads about 10 MB, switching back is instant. `?model=v2.0` selects one, `?model=<url>` loads another ONNX file (same input/output shape); `?image=<url>` runs once on a picture.
* The files in `models/` are web copies of the release models with the weights stored at half precision (5.4 MB instead of 10.7 MB; the browser converts them back to full precision when it loads them, and the scores differ by less than 0.01). The page downloads only the model you pick, shows the progress, and keeps it on the device so it downloads once. If you replace a model file, change `REV` at the top of `index.html` so devices fetch the new one.
* To add a model, put the `.onnx` export in `models/` and add it to the `MODELS` list at the top of the script in `index.html`. Ultralytics YOLO11 is AGPL-3.0.
* Speed: one thread of WASM, so a few frames per second on a phone. The Android app uses LiteRT and is faster.
