"""Make a cleaned copy of the Roboflow 'paper' export: dataset/ -> dataset_fixed/.

Problems found in the export (checked by drawing the boxes on the pictures):
  * The 4032x3024 phone photos (about half of the data) lost their EXIF "which way up" tag in the export, but the
    labels belong to the upright picture. Some photos were taken with the phone in portrait (they must be turned 90
    degrees clockwise, or counter-clockwise, to match their labels), others in landscape (already right). For each
    photo this script tries the three options and keeps the one where the labelled box is clearly brighter than its surroundings (the paper balls are
    white). Photos where the two options are too close to call are dropped instead of guessed.
  * About 20 small web-stock pictures (white background, shredded-paper piles, pencils) have boxes that are in the
    wrong places or on things that are not paper balls. They are dropped.
  * The 1200x1600 / 1600x1200 photos already match their labels and are kept as they are.
Pictures are shrunk to at most 1280 px on the long side so training is not slowed down by 12-megapixel files.

Run:  py fix_dataset.py      (needs opencv-python, numpy and Pillow)
"""
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SRC, DST = HERE / "dataset", HERE / "dataset_fixed"
GOOD_SIZES = {(1200, 1600), (1600, 1200)}      # WhatsApp-style photos: labels already fit
PHONE_SIZE = (4032, 3024)                       # phone photos: orientation has to be worked out
MIN_MARGIN = 12.0                               # the best orientation must beat the second best by this much...
MIN_SCORE = 25.0                                # ...and the box must be at least this much brighter than its surroundings
MIN_ASPECT, MAX_ASPECT = 0.5, 2.0               # box width/height (in pixels) a paper ball can plausibly have
MAX_SIDE = 1280
MIN_VALID = 30                                  # the export's own validation set is tiny once cleaned: top it up
TURNS = {"raw": None, "cw": cv2.ROTATE_90_CLOCKWISE, "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE}


def read_boxes(label_path):
    rows = [r.split() for r in label_path.read_text().splitlines() if r.strip()]
    return [tuple(map(float, r[1:5])) for r in rows]


def box_contrast(img, boxes):
    """How much brighter the labelled boxes are than the ring around them (higher = the boxes sit on the white ball)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    values = []
    for x, y, bw, bh in boxes:
        x1, y1 = max(int((x - bw / 2) * w), 0), max(int((y - bh / 2) * h), 0)
        x2, y2 = min(int((x + bw / 2) * w), w), min(int((y + bh / 2) * h), h)
        if x2 - x1 < 4 or y2 - y1 < 4:
            values.append(-99.0)
            continue
        inner = gray[y1:y2, x1:x2].mean()
        m = int(0.5 * max(x2 - x1, y2 - y1))
        X1, Y1, X2, Y2 = max(x1 - m, 0), max(y1 - m, 0), min(x2 + m, w), min(y2 + m, h)
        ring_area = (Y2 - Y1) * (X2 - X1) - (y2 - y1) * (x2 - x1)
        ring = (gray[Y1:Y2, X1:X2].sum() - gray[y1:y2, x1:x2].sum()) / max(ring_area, 1)
        values.append(inner - ring)
    return float(np.mean(values))


def shrink(img):
    scale = MAX_SIDE / max(img.shape[:2])
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img


def main():
    if not SRC.is_dir():
        raise SystemExit(f"{SRC} not found: unzip the Roboflow export there first")
    if DST.exists():
        for child in DST.iterdir():   # clear the contents, not the folder itself (Windows refuses if a shell is inside it)
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    kept, dropped = {}, {"stock": 0, "unsure": 0}
    turned = {name: 0 for name in TURNS}
    for split in ("train", "valid", "test"):
        (DST / split / "images").mkdir(parents=True)
        (DST / split / "labels").mkdir(parents=True)
        for img_path in sorted((SRC / split / "images").glob("*.*")):
            label = SRC / split / "labels" / (img_path.stem + ".txt")
            size = Image.open(img_path).size
            if not label.is_file() or size not in GOOD_SIZES | {PHONE_SIZE}:
                dropped["stock"] += 1
                continue
            if size == PHONE_SIZE:
                small = cv2.imread(str(img_path), cv2.IMREAD_REDUCED_COLOR_4)
                boxes = read_boxes(label)
                scores = {name: box_contrast(small if turn is None else cv2.rotate(small, turn), boxes)
                          for name, turn in TURNS.items()}
                ranked = sorted(scores, key=scores.get, reverse=True)
                choice = ranked[0]
                if scores[choice] - scores[ranked[1]] < MIN_MARGIN or scores[choice] < MIN_SCORE:
                    dropped["unsure"] += 1
                    continue
                frame_w, frame_h = PHONE_SIZE if choice == "raw" else PHONE_SIZE[::-1]
                if any(not MIN_ASPECT <= bw * frame_w / (bh * frame_h) <= MAX_ASPECT for _, _, bw, bh in boxes):
                    dropped["unsure"] += 1   # a very thin or very flat box is not a paper ball: wrong orientation
                    continue
                turned[choice] += 1
                img = cv2.imread(str(img_path))
                if TURNS[choice] is not None:
                    img = cv2.rotate(img, TURNS[choice])
            else:
                img = cv2.imread(str(img_path))
            cv2.imwrite(str(DST / split / "images" / (img_path.stem + ".jpg")), shrink(img), [cv2.IMWRITE_JPEG_QUALITY, 92])
            shutil.copy2(label, DST / split / "labels" / label.name)
            kept[split] = kept.get(split, 0) + 1
    valid_images = list((DST / "valid" / "images").glob("*.jpg"))
    if len(valid_images) < MIN_VALID:     # move a seeded random sample of training pictures into validation
        train_images = sorted((DST / "train" / "images").glob("*.jpg"))
        for img in random.Random(0).sample(train_images, MIN_VALID - len(valid_images)):
            shutil.move(str(img), DST / "valid" / "images" / img.name)
            shutil.move(str(DST / "train" / "labels" / (img.stem + ".txt")), DST / "valid" / "labels" / (img.stem + ".txt"))
        kept["train"] -= MIN_VALID - len(valid_images)
        kept["valid"] = MIN_VALID
    (DST / "data.yaml").write_text("train: train/images\nval: valid/images\nnc: 1\nnames: ['paper']\n")
    print(f"kept {kept}; dropped {dropped}; phone photos as-is / turned cw / turned ccw: {turned['raw']} / {turned['cw']} / {turned['ccw']}")
    print(f"-> {DST}")


if __name__ == "__main__":
    main()
