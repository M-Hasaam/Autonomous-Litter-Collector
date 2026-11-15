"""Download "no paper ball here" negatives of OTHER balls from the Hugging Face dataset AIOmarRehan/Sports-Balls (CC0).

15 ball types (football, basketball, tennis, golf, table tennis, volleyball, ...), a few images of each, resized to <= 640 px and
written with EMPTY label files (YOLO "background" images) into dataset/{train,valid}/negative/ plus lines in dataset/SOURCES/negative.txt.
Uses the public datasets-server rows API (no pyarrow / datasets library needed). Look at the images before training on them.

Run:  python fetch_hf_balls.py [--per-class 7]
"""
import argparse
import io
import json
import random
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
DATASET = "AIOmarRehan/Sports-Balls"
API = f"https://datasets-server.huggingface.co/rows?dataset={DATASET}&config=default&split=test"
MAX_SIDE, VALID_FRACTION = 640, 0.15


def get(url):
    for attempt in range(8):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == 7:
                raise
            time.sleep(10 * (attempt + 1))   # 429 = asked too fast: back off and retry


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--per-class", type=int, default=7)
    ap.add_argument("--out", type=Path, default=HERE / "dataset")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows, offset, names, urls = [], 0, None, {}
    while True:
        page = json.loads(get(f"{API}&offset={offset}&length=100"))
        names = names or page["features"][1]["type"]["names"]
        rows += [(r["row_idx"], names[r["row"]["label"]]) for r in page["rows"]]
        urls.update({r["row_idx"]: r["row"]["image"]["src"] for r in page["rows"]})   # signed image links, valid for a while
        time.sleep(1)
        if len(page["rows"]) < 100:
            break
        offset += 100
    by_class = defaultdict(list)
    for idx, label in rows:
        by_class[label].append(idx)
    chosen = []
    for label, idxs in sorted(by_class.items()):
        rng.shuffle(idxs)
        chosen += [(label, i) for i in idxs[:args.per_class]]
        print(f"{label}: {len(idxs)} available, taking {min(len(idxs), args.per_class)}")
    rng.shuffle(chosen)
    n_valid = round(len(chosen) * VALID_FRACTION)
    sources, saved = [], 0
    for k, (label, idx) in enumerate(chosen):
        stem = f"hfball_{idx:04d}"
        if any(args.out.glob(f"*/negative/images/{stem}.jpg")):
            continue   # already downloaded by an earlier run
        img = None
        for attempt in range(6):
            try:
                img = Image.open(io.BytesIO(get(urls[idx]))).convert("RGB")
                break
            except Exception as e:
                time.sleep(3 * (attempt + 1))   # the server answers 429 when asked too fast: wait and retry
        if img is None:
            print("skip", idx)
            continue
        time.sleep(0.5)
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        split = "valid" if k < n_valid else "train"
        for sub in ("images", "labels"):
            (args.out / split / "negative" / sub).mkdir(parents=True, exist_ok=True)
        img.save(args.out / split / "negative" / "images" / f"{stem}.jpg", quality=90)
        (args.out / split / "negative" / "labels" / f"{stem}.txt").write_text("", encoding="utf-8")
        sources.append(f"{split}/{stem}.jpg\t{label}\tCC0 1.0\tHugging Face {DATASET}\thttps://huggingface.co/datasets/{DATASET}")
        saved += 1
    (args.out / "SOURCES").mkdir(exist_ok=True)
    with open(args.out / "SOURCES" / "negative.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(sources) + "\n")
    print(f"saved {saved} images to {args.out}")


if __name__ == "__main__":
    main()
