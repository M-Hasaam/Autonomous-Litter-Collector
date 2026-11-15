# Autonomous Litter Collector (LitterVision)

A small robot that finds litter with a camera and a YOLO model: an ESP32-CAM / ESP32 Wi-Fi bridge for the camera, an ESP32-S3 car with
servo arm, PC tools and an Android app that runs the detector on the phone. The first target is **crumpled paper balls on the floor**.

## Try the paper-ball detector in your browser

**https://m-hasaam.github.io/Autonomous-Litter-Collector/paperball-test/**

Open it on a phone (Safari or Chrome), tap *Start camera* and point it at a paper ball, then at things that are not paper balls
(other balls, socks, bags, white walls). The model runs **inside your browser, on your device**; nothing is uploaded. "Pick photo"
tests a single picture and the slider shows weaker detections. It runs the v2.0 model. Source: [docs/paperball-test/](docs/paperball-test/).

## Paper-ball detector: results

**Current model: v2.0** ([release](https://github.com/M-Hasaam/Autonomous-Litter-Collector/releases/tag/v2.0), `roboflow-paperball.pt` / `.onnx`).
YOLO11n fine-tuned for 80 epochs (best epoch 70) on 697 training pictures: the cleaned Roboflow set, 271 pictures **without** paper balls,
low-angle floor frames and real desk frames (see the dataset below). Scores on pictures the model never trained on:

| Set | Pictures | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| valid | 124 | 0.987 | 1.000 | 0.995 | 0.780 |
| test1 (same kind as the training data) | 25 | 1.000 | 0.997 | 0.995 | 0.799 |
| test2 (real ESP32-CAM, hand-labelled) | 23 | 1.000 | 0.913 | 0.992 | 0.855 |
| lowcam_front (low-angle floor camera) | 29 | 1.000 | 0.998 | 0.995 | 0.829 |
| agilex_desk (desk frames, automatic labels) | 11 | 0.993 | 1.000 | 0.995 | 0.734 |

**False detections.** v1 drew a box on other white or bright surfaces. On 48 held-out pictures with no paper ball (balls, round food, bags,
floors, walls) at confidence 0.25: **v1 gave false boxes on 38 of 48, v2 on 1 of 48** (0 at confidence 0.5).

Predictions (green = label, red = model):

![test2 predictions](Training/paper_ball_detector/results/v2/test2.png)
![test1 predictions](Training/paper_ball_detector/results/v2/test1.png)
![lowcam_front predictions](Training/paper_ball_detector/results/v2/lowcam_front.png)

**Limitations.** The test sets are small and almost saturated (about 0.995), so they cannot separate close models; the false-detection check above
uses internet pictures, not your own floor; very blurry frames can still be missed (test2 recall 0.913); agilex_desk labels are automatic.
Collecting real-camera pictures, including ones without a paper ball, is the next step. Scores, sheets and the per-epoch log are in
`Training/paper_ball_detector/results/v2/`.

**Also tried:** adding 120 synthetic "paper ball next to other balls" pictures to training (`data_with_confusion.yaml`). It did not help on the real-camera
tests (test2 mAP50-95 0.731 vs 0.855, 2 vs 1 pictures with a false box), so v2 does not use them.

**v1.0** ([release](https://github.com/M-Hasaam/Autonomous-Litter-Collector/releases/tag/v1.0)): trained on the Roboflow set only (248 train / 30 valid),
mAP50 0.995 valid / 0.939 test1 / 0.915 test2 on the original, uncorrected test sets (`results/v1/`). Its weakness was the false detections above.

## Dataset (`Training/paper_ball_detector/dataset/`)

Layout: `dataset/{train,valid,test}/<source>/{images,labels}`, YOLO format, one class (`paper`). `data.yaml` lists the real data,
`data_with_confusion.yaml` adds the synthetic pictures to training only. Licences and notes: `dataset/SOURCES/`.

| Source | train | valid | test | What it is |
|---|---|---|---|---|
| `normal` / `test1` | 248 | 30 | 25 | cleaned Roboflow "paper" set (internet and phone photos), CC BY 4.0 |
| `negative` | 271 | 48 | - | pictures **without** paper balls (empty labels): balls, round food, bags, floors, walls...; Open Images (CC BY 2.0) and Hugging Face Sports-Balls (CC0) |
| `lowcam_front` | 140 | 30 | 29 | low-angle floor frames, Roboflow project paper-ball-detection, CC BY 4.0, split in blocks of consecutive frames |
| `agilex_desk` | 38 | 16 | 11 | real desk frames from the robot dataset agilex_gather_paper_ball (Apache-2.0); **automatic labels** (background subtraction), split by episode |
| `confusion` | 120 | 24 | - | **synthetic**: real paper balls pasted onto other-ball backgrounds; training only, never test data |
| `test2` | - | - | 23 | real ESP32-CAM pictures, labelled by hand; the main score for the real camera |

Test sets are never used for training. A zip of the whole folder (`dataset.zip`, for the Colab notebook) is built locally and not committed.

## Train, evaluate, export

All in `Training/paper_ball_detector/`:

| Task | How |
|---|---|
| Train on Colab (GPU) | `train_paperball_colab.ipynb`; upload `dataset.zip`; set `USE_CONFUSION` to `False` or `True` |
| Train locally | `python train_roboflow.py [--epochs 25] [--device cpu]` (needs ultralytics, torch, onnx) |
| Score a model | `python evaluate_on_camera.py --set all` (test1, test2, lowcam_front, agilex_desk) |
| Put a model in the Android app | `python export_model.py --weights roboflow-paperball.pt --install` |
| Rebuild data | `fetch_negatives.py`, `fetch_hf_balls.py`, `make_confusion.py`, `import_agilex.py` |
| Try the model in the browser | replace the `.onnx` in `docs/paperball-test/` with the new export, push |

## Repository layout

```
Mobile_App/            Android app (Kotlin, Compose, CameraX, LiteRT); open this folder in Android Studio
Embedded/
  dashboard.py         PC dashboard: camera (ESP32 Wi-Fi bridge) + car (ESP32-S3 over Bluetooth LE)
  esp32/WifiBridge/    ESP32 camera bridge firmware + PC viewers (camera_yolo, camera_distance, camera_serial)
  esp32-s3/AI_Car/     ESP32-S3 car (BLE, 4 servos): AI_Car.ino and control_car.py
  Other/               older Arduino sketches and the ESP32-CAM web server
Training/
  paper_ball_detector/ dataset, trainer, Colab notebook, evaluation, export, results, trained model
docs/
  paperball-test/      the browser test page (published with GitHub Pages)
  MeArm-with-MG996R/   robot-arm drawings and datasheets
```

## Android app status

The app runs a YOLO model fully on the phone (CameraX + LiteRT, no cloud). The model currently installed in
`Mobile_App/app/src/main/assets/` is the **generic pretrained YOLO11n (COCO classes), not the paper-ball model**. Install the paper-ball
model with `export_model.py --install` (see above). The app has been built, but not yet tested on a device.

## Licences

Code in this repository: see the repository owner. Models fine-tuned from Ultralytics YOLO11 are **AGPL-3.0**. Data: Roboflow Universe
sets are CC BY 4.0 (credit the authors if you share the data or models), Open Images pictures CC BY 2.0, Sports-Balls CC0,
agilex_gather_paper_ball Apache-2.0. Per-source notes are in `Training/paper_ball_detector/dataset/SOURCES/`.
