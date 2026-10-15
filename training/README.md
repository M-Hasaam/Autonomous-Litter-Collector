# LitterVision training workspace

Fine-tune a YOLO11n object detector on your own labeled litter photos and put the result
into the Android app. Nothing here is trained yet: the app currently runs the **pretrained
Ultralytics YOLO11n** model, which only knows the 80 general COCO classes (person, bottle,
cup, ...). It is **not** a litter detector. Litter classes (`crumpled_paper`, `can`,
`plastic_bottle`) exist only after you fine-tune.

## 1. Setup (once)

Python 3.10 - 3.12 is recommended. From this `training/` folder:

```powershell
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

A GPU is optional but makes training far faster. Without one, use a small `--epochs` and
`--batch` first, or run the same commands in Google Colab.

## 2. Prepare a labeled dataset (YOLO detection format)

1. Take or collect photos of the three classes: `crumpled_paper`, `can`, `plastic_bottle`.
   Aim for at least a few hundred varied images (different light, backgrounds, distances).
2. Label them with a bounding-box tool that can export YOLO format (for example
   Roboflow, CVAT or Label Studio). Each image gets a `.txt` file with the same name:

   ```
   <class_id> <x_center> <y_center> <width> <height>     # all values 0..1
   ```

   Class ids follow `dataset.yaml.example`: `0 crumpled_paper`, `1 can`, `2 plastic_bottle`.
3. Arrange the files and keep some images (about 20%) for validation:

   ```
   my_litter_dataset/
       images/train/  *.jpg        labels/train/  *.txt
       images/val/    *.jpg        labels/val/    *.txt
   ```
4. Copy `dataset.yaml.example` to `my_dataset.yaml` and edit `path` to point at your dataset
   folder. Do not change the class order after labeling.

Datasets, runs and `.pt` files are git-ignored; do not commit them.

## 3. Train and validate

```powershell
python train.py my_dataset.yaml --epochs 100
```

`train.py` checks the YAML and folders first, fine-tunes from `yolo11n.pt`, then validates.
Output goes to `runs/litter_yolo11n/` (best weights: `runs/litter_yolo11n/weights/best.pt`).
Useful options: `--batch 8` (less memory), `--imgsz 640`, `--device 0`, `--name my_run`.

## 4. Predict on images

```powershell
python predict.py --weights runs/litter_yolo11n/weights/best.pt --source path\to\photos
```

Annotated copies are written to `predictions/predict/`. Run these on photos that were **not**
in the training set to judge real-world quality before putting a model in the app.

## 5. Export to the app

The app loads two files from `app/src/main/assets/`: `litter_detector.tflite` and
`labels.txt`. `export_model.py` produces both in the exact format the app decodes
(float32, input `[1, S, S, 3]` RGB 0..1, output `[1, 4+classes, N]`, no built-in NMS).

```powershell
# the pretrained model (what the app ships with now)
python export_model.py --install

# a fine-tuned model
python export_model.py --weights runs/litter_yolo11n/weights/best.pt --install
```

Or do train + validate + export + install in one go (the app model is replaced only if every
step succeeds, otherwise the previous model is kept):

```powershell
python train.py my_dataset.yaml --epochs 100 --export-to-app
```

Without `--install` the files are exported to a temp folder (or `--out-dir`) and the app is
left untouched. After installing, rebuild and run the app. `--imgsz 320` exports a model that
is roughly 4x faster, at some cost in accuracy for small objects; the app reads the input
size from the model, so no code change is needed. Train with the same `--imgsz` you export.

## Licensing

- **Ultralytics YOLO11 and the `ultralytics` package** are licensed under **AGPL-3.0**
  (an Enterprise License is available from Ultralytics). Models trained or exported with them
  are derivative works; distributing an app that bundles them has AGPL obligations, including
  offering the corresponding source. Read https://www.ultralytics.com/license before
  distributing the app.
- **Pretrained weights** (`yolo11n.pt`) were trained on COCO. COCO annotations are CC BY 4.0
  and its images carry their Flickr licenses.
- **Your own training images** keep whatever license they came with; check it, and any
  dataset you download (e.g. TACO is CC BY 4.0) before use.
- **TensorFlow/LiteRT** (Apache-2.0), **onnx2tf** (MIT) and **ONNX** (Apache-2.0) are used for
  export and on-device inference.

This repository's own license is not changed by any of the above.
