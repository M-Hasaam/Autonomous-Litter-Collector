"""Import real paper-ball frames from the Hugging Face robot dataset villekuosmanen/agilex_gather_paper_ball (Apache-2.0).

The 20 episodes (camera cam_high, fixed) show crumpled paper balls on a white desk in front of a keyboard and monitor while a robot
gripper picks them up: white paper on a white surface with distractors - the situation the detector gets wrong. The videos have no
labels, and the v1 model is NOT a good labeller here (it fires on the same distractors we want to teach it about), so the boxes come
from BACKGROUND SUBTRACTION: the camera never moves and every episode ends with an empty desk, so the median of the last frames is
the background and the paper balls are what differs from it (bright, ball-sized blobs on the desk).
A frame is kept only when the boxes are unambiguous: no robot arm in the picture at all, no blob that looks like two touching balls, no tiny or
huge blobs. Brown crumpled paper is labelled as paper too. Also keeps EMPTY_PER_EPISODE empty-desk frames (empty labels).
Split by episode (never by frame) so near-identical frames do not leak between train / valid / test.
THE LABELS ARE AUTOMATIC: look at the pictures before training.

Output: dataset/{train,valid,test}/agilex_desk/{images,labels} + dataset/SOURCES/agilex_desk.txt
Run:  python import_agilex.py --videos <folder with ep000000.mp4 ... ep000019.mp4>      (needs av, opencv-python, numpy)
Videos: https://huggingface.co/datasets/villekuosmanen/agilex_gather_paper_ball  videos/chunk-000/observation.images.cam_high/
"""
import argparse
from pathlib import Path

import av
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset"
STEP, PER_EPISODE, EMPTY_PER_EPISODE = 15, 8, 1
SPLITS = {"train": range(0, 14), "valid": range(14, 17), "test": range(17, 20)}
DESK_Y = (150, 340)                  # ball centres lie on the desk strip (pixels of the 480-high frame); lower = exposure flicker
MIN_W, MIN_H, MAX_W, MAX_H, MIN_AREA = 40, 35, 220, 190, 900


def find_boxes(img, bg):
    """(boxes, ok): ball boxes by background subtraction; ok=False if the picture is ambiguous (arm near a ball, merged balls)."""
    diff = np.abs(cv2.GaussianBlur(img, (0, 0), 2).astype(np.int16) - cv2.GaussianBlur(bg, (0, 0), 2).astype(np.int16)).max(2) > 45
    mask = (diff & (img.max(2) > 125)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    dark = (img.max(2) < 70) & diff                          # the black robot arm (not the black keyboard: that is in the background)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    boxes, ok = [], dark.mean() < 0.004                      # any visible robot arm = motion blur, shadows, held paper: skip the frame
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if not DESK_Y[0] <= y + h / 2 <= DESK_Y[1] or area < MIN_AREA // 3:
            continue                                         # not on the desk strip: arm parts, flicker, shadows
        if w < MIN_W or h < MIN_H or area < MIN_AREA or w > MAX_W or h > MAX_H or not 0.45 < w / h < 2.4:
            ok = False                                       # a blob that is not clearly one ball
            continue
        if w > 1.3 * h and w > 120:
            ok = False                                       # probably two touching balls: cannot label with one box
            continue
        x1, y1, x2, y2 = max(x - 25, 0), max(y - 25, 0), x + w + 25, y + h + 25
        if dark[y1:y2, x1:x2].mean() > 0.06:
            ok = False                                       # the arm touches / hides the ball
            continue
        boxes.append((x, y, w, h))
    return boxes, ok


def spread(items, n):
    return items if len(items) <= n else [items[int(i)] for i in np.linspace(0, len(items) - 1, n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--videos", type=Path, required=True)
    args = ap.parse_args()
    notes = []
    for split, episodes in SPLITS.items():
        for sub in ("images", "labels"):
            d = DATA / split / "agilex_desk" / sub
            d.mkdir(parents=True, exist_ok=True)
            for old in d.glob("*"):
                old.unlink()
        for ep in episodes:
            frames = [(i, f.to_ndarray(format="bgr24"))
                      for i, f in enumerate(av.open(str(args.videos / f"ep{ep:06d}.mp4")).decode(video=0)) if i % STEP == 0]
            bg = np.median(np.stack([f for _, f in frames[-4:]]), 0).astype(np.uint8)
            hits, empties = [], []
            for i, img in frames[:-4]:
                boxes, ok = find_boxes(img, bg)
                if ok and boxes:
                    hits.append((i, img, boxes))
                elif ok and (np.abs(img.astype(np.int16) - bg).max(2) > 45).mean() < 0.004:
                    empties.append((i, img, []))
            for i, img, boxes in spread(hits, PER_EPISODE) + spread(empties[-3:], EMPTY_PER_EPISODE):
                h, w = img.shape[:2]
                stem = f"agilex_ep{ep:02d}_f{i:04d}"
                cv2.imwrite(str(DATA / split / "agilex_desk" / "images" / (stem + ".jpg")), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                (DATA / split / "agilex_desk" / "labels" / (stem + ".txt")).write_text(
                    "".join(f"0 {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}\n" for x, y, bw, bh in boxes))
                notes.append(f"{split}/{stem}.jpg\t{len(boxes)} box(es)")
            print(split, "episode", ep, f"{len(hits)} usable frames")
    (DATA / "SOURCES" / "agilex_desk.txt").write_text(
        "Frames from villekuosmanen/agilex_gather_paper_ball (Apache-2.0), camera cam_high, https://huggingface.co/datasets/"
        "villekuosmanen/agilex_gather_paper_ball\nLabels are AUTOMATIC (background subtraction, see import_agilex.py) - review before use. "
        "Split by episode: 0-13 train, 14-16 valid, 17-19 test.\n\nfile\tlabels\n" + "\n".join(notes) + "\n", encoding="utf-8")
    print(f"-> {DATA}/<train|valid|test>/agilex_desk")


if __name__ == "__main__":
    main()
