"""Train YOLO11n on a paper-ball dataset downloaded from Roboflow Universe (YOLOv11 / YOLOv8 export), separately from
yolo-temp. It does NOT touch the app's model (Mobile_app/app/src/main/assets/litter_detector.tflite).

The data is the cleaned dataset in dataset/ (see dataset/data.yaml). It started as a Roboflow Universe "paper" export
(CC BY 4.0); the raw export and the script that cleaned it were removed once the cleaned copy was committed.

Run (in an environment with ultralytics + torch + onnx):
  python train_roboflow.py [--epochs 25] [--batch 16] [--device cpu]
Output: roboflow-paperball.pt and roboflow-paperball.onnx in this folder. Try it on the camera with:
  py Embedded/esp32/WifiBridge/scripts/camera_distance.py --model Training/paper_ball_detector/roboflow-paperball.onnx --rotate ccw
"""
import argparse
import shutil
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BASE_WEIGHTS = HERE.parent / "yolo11n.pt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_split(yaml_dir, entry):
    """Roboflow writes entries like '../train/images' that do not match the unzipped layout; try the sensible ones."""
    candidates = [yaml_dir / entry, yaml_dir / entry.replace("../", ""), yaml_dir / Path(entry).name]
    for c in candidates:
        c = c.resolve()
        if c.is_dir():
            return c
    return None


def prepare_yaml(data_yaml, out_path):
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = cfg.get("names")
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names or [])
    if not names:
        sys.exit("ERROR: data.yaml has no class names")
    resolved = {"path": str(data_yaml.parent.resolve()), "names": {i: n for i, n in enumerate(names)}}
    print(f"classes: {names}")
    for split in ("train", "val", "test"):
        entry = cfg.get(split) or ({"val": cfg.get("valid")}.get(split))
        if not entry:
            continue
        entries = entry if isinstance(entry, list) else [entry]
        folders = [find_split(data_yaml.parent, str(e)) for e in entries]
        if any(f is None for f in folders):
            print(f"WARNING: cannot find the '{split}' images folder ({entry})")
            folders = [f for f in folders if f is not None]
            if not folders:
                continue
        count = sum(1 for folder in folders for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        print(f"{split}: {count} images in {', '.join(str(f) for f in folders)}")
        resolved[split] = str(folders[0]) if len(folders) == 1 else [str(f) for f in folders]
    if "train" not in resolved or "val" not in resolved:
        sys.exit("ERROR: need both a train and a validation (valid) folder")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(HERE / "dataset" / "data.yaml"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.is_file():
        sys.exit(f"ERROR: {data_yaml} not found. Expected the cleaned dataset in {HERE / 'dataset'} "
                 f"(see the top of this file).")
    resolved = HERE / "runs" / "dataset.resolved.yaml"
    prepare_yaml(data_yaml, resolved)

    from ultralytics import YOLO

    model = YOLO(str(BASE_WEIGHTS))
    results = model.train(data=str(resolved), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                          device=args.device, project=str(HERE / "runs"), name="roboflow-paperball", exist_ok=True,
                          workers=0, flipud=0.0, plots=False)
    best = Path(results.save_dir) / "weights" / "best.pt"
    shutil.copy2(best, HERE / "roboflow-paperball.pt")
    metrics = YOLO(str(HERE / "roboflow-paperball.pt")).val(data=str(resolved), imgsz=args.imgsz, workers=0)
    print(f"\nValidation mAP50 {metrics.box.map50:.3f}, mAP50-95 {metrics.box.map:.3f}")
    onnx = Path(YOLO(str(HERE / "roboflow-paperball.pt")).export(format="onnx", imgsz=args.imgsz, simplify=True, opset=12))
    if onnx.resolve() != (HERE / "roboflow-paperball.onnx").resolve():
        shutil.copy2(onnx, HERE / "roboflow-paperball.onnx")
    print(f"Saved {HERE / 'roboflow-paperball.pt'} and roboflow-paperball.onnx")


if __name__ == "__main__":
    main()
