"""Download "no paper ball here" negative images from Open Images (validation split, CC BY).

Open Images image-level labels are used to pick pale / round / cluttered things that a paper-ball
detector tends to confuse with paper (plastic bags, eggs, sheets, walls, floors, ...). Any image that
also carries a Paper / Book / Tissue / Napkin label is skipped. Images are resized to <= 640 px and
written with EMPTY label files (YOLO "background" images), plus SOURCES/negative.txt with author + licence.

Output (review it before using!):  dataset/{train,valid}/negative/{images,labels}/ and dataset/SOURCES/negative.txt

Run:  python fetch_negatives.py --meta <folder with the 3 Open Images CSVs>
The CSVs are  oidv7-class-descriptions.csv, oidv7-val-annotations-human-imagelabels.csv  (storage.googleapis.com/openimages/v7/)
and  validation-images-with-rotation.csv  (storage.googleapis.com/openimages/2018_04/validation/).
Needs Pillow. Open Images images are CC BY 2.0 (Flickr); keep SOURCES/negative.txt for attribution.
"""
import argparse
import csv
import io
import random
import urllib.request
from collections import defaultdict
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent

# class -> how many images to take (hard negatives first: pale/round things look like paper balls)
QUOTAS = {
    "Plastic bag": 16, "Egg": 8, "Golf ball": 6, "Tennis ball": 6, "Cotton": 8, "Sock": 8,
    "Tablecloth": 10, "Bed sheet": 12, "Pillow": 8, "Towel": 8, "Curtain": 8, "Wall": 12,
    "Floor": 16, "Tile": 8, "Carpet": 6, "Rug": 5, "Laptop": 6, "Computer keyboard": 6,
    "Cable": 6, "Rock": 4, "Pebble": 4,
}
# second round (--extra): bigger balls and other round / crumpled / pale things a paper-ball detector may confuse with paper
EXTRA_QUOTAS = {
    "Ball (Object)": 14, "Football": 10, "Rugby ball": 6, "Volleyball (Ball)": 6, "Balloon": 8, "Snowman": 3,
    "Garlic": 4, "Onion": 4, "Cabbage": 4, "Cauliflower": 4, "Coconut": 3, "Pearl": 2, "Dumpling": 4,
    "Ice cream": 4, "Bowl": 6, "Mug": 5, "Lampshade": 5, "Light fixture": 5, "Candle": 4, "Aluminum can": 4,
}
EXCLUDE = {"Paper", "Book", "Paper towel", "Tissue paper", "Napkin"}
MAX_SIDE = 640
VALID_FRACTION = 0.15


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE / "dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra", action="store_true",
                    help="second round: use EXTRA_QUOTAS, skip images already in --out, append to SOURCES/negative.txt")
    args = ap.parse_args()
    quotas = EXTRA_QUOTAS if args.extra else QUOTAS
    have = {p.stem for p in args.out.glob("*/negative/images/*.jpg")} if args.extra else set()
    rng = random.Random(args.seed)

    ids = {}
    with open(args.meta / "oidv7-class-descriptions.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ids[row["DisplayName"]] = row["LabelName"]
    code_to_name = {ids[n]: n for n in list(quotas) + list(EXCLUDE) if n in ids}

    by_class, bad = defaultdict(set), set()
    with open(args.meta / "oidv7-val-annotations-human-imagelabels.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = code_to_name.get(row["LabelName"])
            if name is None or row["Confidence"] != "1":
                continue
            if name in EXCLUDE:
                bad.add(row["ImageID"])
            else:
                by_class[name].add(row["ImageID"])

    info = {}
    with open(args.meta / "validation-images-with-rotation.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            info[row["ImageID"]] = row

    chosen, seen = [], set()
    for name, quota in quotas.items():
        pool = sorted(i for i in by_class[name] if i not in bad and i not in seen and i in info and f"neg_{i}" not in have)
        rng.shuffle(pool)
        taken = 0
        for image_id in pool:
            if taken >= quota:
                break
            chosen.append((name, image_id))
            seen.add(image_id)
            taken += 1
        print(f"{name}: {taken}/{quota} candidates")

    rng.shuffle(chosen)
    n_valid = round(len(chosen) * VALID_FRACTION)
    sources, saved = [], 0
    for idx, (name, image_id) in enumerate(chosen):
        row = info[image_id]
        data = None
        mirror = f"https://open-images-dataset.s3.amazonaws.com/validation/{image_id}.jpg"  # Flickr originals are mostly gone
        for url in (mirror, row["OriginalURL"], row["Thumbnail300KURL"]):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=20) as r:
                    data = r.read()
                break
            except Exception:
                continue
        if data is None:
            continue
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            continue
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        split = "valid" if idx < n_valid else "train"
        (args.out / split / "negative" / "images").mkdir(parents=True, exist_ok=True)
        (args.out / split / "negative" / "labels").mkdir(parents=True, exist_ok=True)
        stem = f"neg_{image_id}"
        img.save(args.out / split / "negative" / "images" / f"{stem}.jpg", quality=90)
        (args.out / split / "negative" / "labels" / f"{stem}.txt").write_text("", encoding="utf-8")
        sources.append(f"{split}/{stem}.jpg\t{name}\t{row['License']}\t{row['Author']}\t{row['OriginalLandingURL']}")
        saved += 1
    (args.out / "SOURCES").mkdir(exist_ok=True)
    sources_file = args.out / "SOURCES" / "negative.txt"
    if args.extra and sources_file.is_file():      # second round: add to the attribution list, never overwrite it
        with open(sources_file, "a", encoding="utf-8") as f:
            f.write("\n".join(sources) + "\n")
    else:
        sources_file.write_text("file\tsearch_class\tlicense\tauthor\tsource_page\n" + "\n".join(sources) + "\n", encoding="utf-8")
    print(f"saved {saved} images to {args.out}")


if __name__ == "__main__":
    main()
