"""Build SYNTHETIC "confusion" training pictures: real paper balls pasted onto pictures of other balls / round / white things.

Why: the detector must learn that a paper ball next to a football, an egg or a white sheet is still a paper ball, and the football
is not. Real photos like that are rare, so each picture is made by cutting a labelled paper ball out of the cleaned dataset (soft
elliptical edge) and pasting it, scaled / flipped / brightness-shifted, onto a background from negatives_internet/. Only the pasted
paper balls are labelled; the other balls stay unlabelled, so they act as hard negatives in the same picture.
Leakage is avoided: train composites use train backgrounds + train paper balls, valid composites use valid ones only.
These are fakes (no matching shadows) - treat them as extra hard-negative training data, never as test data, and replace them with
real photos of paper balls among other balls when you collect them.

Output: dataset_fixed/confusion_composite/{train,valid}/{images,labels} + SOURCE.txt
Run:  python make_confusion.py [--train 120 --valid 24]      (needs opencv-python and numpy)
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DATA, NEG = HERE / "dataset_fixed", HERE / "negatives_internet"
OUT = DATA / "confusion_composite"
# backgrounds that look most like paper balls come first; plain surfaces fill up the rest
ROUND = {"Ball (Object)", "Football", "Rugby ball", "Volleyball (Ball)", "Balloon", "Golf ball", "Tennis ball", "Egg", "Cotton",
         "Plastic bag", "Pillow", "Sock", "Garlic", "Onion", "Cabbage", "Cauliflower", "Coconut", "Pearl", "Ice cream", "Bowl",
         "Mug", "Candle", "Rock", "Pebble"}
ROUND_HF = {"american_football", "baseball", "basketball", "billiard_ball", "bowling_ball", "cricket_ball", "football", "golf_ball",
            "hockey_ball", "rugby_ball", "table_tennis_ball", "tennis_ball", "volleyball"}


def read_boxes(path):
    return [tuple(map(float, r.split()[1:5])) for r in path.read_text().splitlines() if r.strip()]


def cutouts(split):
    """Paper-ball crops (with a feathered elliptical alpha) from real, correctly labelled pictures of one split."""
    crops = []
    for lab in sorted((DATA / split / "labels").glob("*.txt")):
        img_path = DATA / split / "images" / (lab.stem + ".jpg")
        img = cv2.imread(str(img_path)) if img_path.is_file() else None
        if img is None:
            continue
        h, w = img.shape[:2]
        for x, y, bw, bh in read_boxes(lab):
            if bw * w < 40 or bh * h < 40 or not 0.6 < bw * w / (bh * h) < 1.7:
                continue   # too small / too thin to make a clean cut-out
            x1, y1, x2, y2 = (max(int((x - bw / 2) * w), 0), max(int((y - bh / 2) * h), 0),
                              min(int((x + bw / 2) * w), w), min(int((y + bh / 2) * h), h))
            crop = img[y1:y2, x1:x2]
            mask = np.zeros(crop.shape[:2], np.float32)
            cv2.ellipse(mask, (crop.shape[1] // 2, crop.shape[0] // 2), (int(crop.shape[1] * 0.46), int(crop.shape[0] * 0.46)),
                        0, 0, 360, 1.0, -1)
            crops.append((crop, cv2.GaussianBlur(mask, (0, 0), max(crop.shape[:2]) * 0.04)))
    return crops


def backgrounds(split):
    classes = {}
    for line in (NEG / "SOURCES.txt").read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].startswith(split + "/"):
            classes[Path(parts[0]).stem] = parts[1]
    paths = sorted((NEG / split / "images").glob("*.jpg"))
    round_first = [p for p in paths if classes.get(p.stem) in ROUND | ROUND_HF]
    return round_first + [p for p in paths if p not in round_first]


def paste(bg, crop, mask, rng):
    h, w = bg.shape[:2]
    cw = rng.uniform(0.10, 0.32) * w                      # ball width as a share of the picture
    scale = cw / crop.shape[1]
    ch = crop.shape[0] * scale
    if ch > 0.6 * h:
        scale *= 0.6 * h / ch
    size = (max(int(crop.shape[1] * scale), 12), max(int(crop.shape[0] * scale), 12))
    c, m = cv2.resize(crop, size, interpolation=cv2.INTER_AREA), cv2.resize(mask, size, interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        c, m = c[:, ::-1], m[:, ::-1]
    c = np.clip(c.astype(np.float32) * rng.uniform(0.8, 1.15) + rng.uniform(-10, 10), 0, 255)
    x0, y0 = rng.randint(0, max(w - size[0], 0)), rng.randint(0, max(h - size[1], 0))
    roi = bg[y0:y0 + size[1], x0:x0 + size[0]].astype(np.float32)
    bg[y0:y0 + size[1], x0:x0 + size[0]] = (roi * (1 - m[..., None]) + c * m[..., None]).astype(np.uint8)
    return (x0 + size[0] / 2) / w, (y0 + size[1] / 2) / h, size[0] / w, size[1] / h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--train", type=int, default=120)
    ap.add_argument("--valid", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    notes = []
    for split, count in (("train", args.train), ("valid", args.valid)):
        for sub in ("images", "labels"):
            d = OUT / split / sub
            d.mkdir(parents=True, exist_ok=True)
            for old in d.glob("*"):
                old.unlink()
        crops, bgs = cutouts(split), backgrounds(split)[:count]
        print(f"{split}: {len(crops)} paper-ball cut-outs, {len(bgs)} backgrounds")
        for i, bg_path in enumerate(bgs):
            bg = cv2.imread(str(bg_path))
            rows = []
            for _ in range(rng.choice([1, 1, 2])):
                crop, mask = rng.choice(crops)
                x, y, w, h = paste(bg, crop, mask, rng)
                rows.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            stem = f"conf_{split}_{i:03d}_{bg_path.stem}"
            cv2.imwrite(str(OUT / split / "images" / (stem + ".jpg")), bg, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (OUT / split / "labels" / (stem + ".txt")).write_text("\n".join(rows) + "\n")
            notes.append(f"{split}/{stem}.jpg\tbackground negatives_internet/{split}/images/{bg_path.name}")
    (OUT / "SOURCE.txt").write_text(
        "SYNTHETIC pictures made by make_confusion.py: real paper-ball cut-outs pasted onto negatives_internet backgrounds "
        "(other balls / round / white things stay unlabelled).\nNot real photos - never use as test data.\n\n" + "\n".join(notes) + "\n",
        encoding="utf-8")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
