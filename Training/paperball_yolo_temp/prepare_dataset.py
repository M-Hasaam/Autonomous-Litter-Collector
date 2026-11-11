"""Build the TEMPORARY 'yolo-temp' dataset from Embedded/esp32-cam/captures/ and boxes.json.

The captures were saved by camera_yolo.py with its detection boxes, labels and header text drawn into the
pixels. Training on them as-is would teach the model to look for green boxes, so the overlay is removed first
(green / yellow overlay pixels are inpainted). The remaining boxes come from boxes.json (hand-drawn).

Run:  py prepare_dataset.py        (needs only opencv-python and numpy)
Output: dataset/images/{train,val}, dataset/labels/{train,val}, dataset/dataset.yaml
"""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
CAPTURES = HERE.parents[1] / "Embedded" / "esp32-cam" / "captures"
DATASET = HERE / "dataset"
VAL_INDEXES = {3, 11, 16, 21}  # position in the sorted list: a top-down view and a near, a middle and a far ball


def remove_overlay(img):
    """Inpaint the green boxes/label bars and the yellow header text that camera_yolo.py draws."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = (h >= 40) & (h <= 85) & (s > 140) & (v > 140)
    yellow = (h >= 22) & (h < 40) & (s > 140) & (v > 160)
    mask = ((green | yellow) * 255).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):  # filled label bars: cover the whole rectangle, which also hides the black text in it
        x, y, w, hh, area = stats[i]
        if hh <= 24 and 20 < w < 170 and area > 0.5 * w * hh:
            mask[max(0, y - 2):y + hh + 2, max(0, x - 2):x + w + 2] = 255
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)


def main():
    boxes = {k: v for k, v in json.loads((HERE / "boxes.json").read_text()).items() if not k.startswith("_")}
    if DATASET.exists():
        shutil.rmtree(DATASET)
    for split in ("train", "val"):
        (DATASET / "images" / split).mkdir(parents=True)
        (DATASET / "labels" / split).mkdir(parents=True)

    counts = {"train": 0, "val": 0}
    for index, name in enumerate(sorted(boxes)):
        src = CAPTURES / name
        if not src.is_file():
            print(f"skip {name}: not in {CAPTURES}")
            continue
        img = remove_overlay(cv2.imread(str(src)))
        h, w = img.shape[:2]
        x1, y1, x2, y2 = boxes[name]
        split = "val" if index in VAL_INDEXES else "train"
        cv2.imwrite(str(DATASET / "images" / split / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        line = f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n"
        (DATASET / "labels" / split / (Path(name).stem + ".txt")).write_text(line)
        counts[split] += 1

    (DATASET / "dataset.yaml").write_text(
        f"path: {DATASET.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: paper_ball\n")  # absolute: Ultralytics resolves '.' unpredictably
    print(f"dataset written to {DATASET}: {counts['train']} train, {counts['val']} val")


if __name__ == "__main__":
    main()
