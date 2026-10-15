"""Fine-tune YOLO11n on a labeled litter dataset (YOLO detection format).

Example:
    python train.py my_dataset.yaml --epochs 100
    python train.py my_dataset.yaml --epochs 100 --export-to-app

Results are saved under training/runs/<name>/ (weights in runs/<name>/weights/best.pt).
With --export-to-app the Android model is replaced ONLY after training, validation and
export all succeed; otherwise the model currently in the app is left untouched.
"""
import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_CLASSES = ["crumpled_paper", "can", "plastic_bottle"]


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def resolve_dataset(yaml_path: Path) -> dict:
    """Validate the dataset YAML and return it with absolute paths."""
    if not yaml_path.is_file():
        fail(f"Dataset YAML not found: {yaml_path}")
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}

    names = data.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    if not names or not isinstance(names, list):
        fail("Dataset YAML must define 'names' (class list).")
    if "nc" in data and data["nc"] != len(names):
        fail(f"'nc' ({data['nc']}) does not match the number of names ({len(names)}).")
    if names != EXPECTED_CLASSES:
        print(
            f"WARNING: class names {names} differ from the planned {EXPECTED_CLASSES}. "
            "Continuing; the exported labels.txt will use your names.",
            file=sys.stderr,
        )

    # Relative 'path' is resolved against the YAML file's folder (not Ultralytics' datasets dir).
    root = Path(data.get("path", "."))
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    if not root.is_dir():
        fail(f"Dataset root folder not found: {root}")

    for split in ("train", "val"):
        entry = data.get(split)
        if not entry:
            fail(f"Dataset YAML is missing '{split}'.")
        for item in entry if isinstance(entry, list) else [entry]:
            location = Path(item)
            if not location.is_absolute():
                location = root / location
            if location.is_dir():
                images = [p for p in location.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES]
                if not images:
                    fail(f"No images found in {split} folder: {location}")
            elif not location.is_file():
                fail(f"'{split}' path does not exist: {location}")

    data["path"] = str(root)
    data["names"] = {i: n for i, n in enumerate(names)}
    data.pop("nc", None)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("dataset", type=Path, help="Path to your dataset YAML (copy of dataset.yaml.example).")
    parser.add_argument("--weights", default="yolo11n.pt", help="Starting weights (default: yolo11n.pt).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16, help="Lower this if you run out of memory.")
    parser.add_argument("--device", default=None, help="e.g. 0 for the first GPU, or cpu.")
    parser.add_argument("--name", default="litter_yolo11n", help="Run folder name under training/runs/.")
    parser.add_argument("--export-to-app", action="store_true",
                        help="After successful train+val, export best.pt and replace the Android app model.")
    args = parser.parse_args()

    data = resolve_dataset(args.dataset.resolve())

    from ultralytics import YOLO

    RUNS_DIR.mkdir(exist_ok=True)
    resolved_yaml = RUNS_DIR / f"{args.name}_dataset.resolved.yaml"
    resolved_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    model = YOLO(args.weights)
    train_kwargs = dict(
        data=str(resolved_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        project=str(RUNS_DIR), name=args.name, exist_ok=False,
    )
    if args.device is not None:
        train_kwargs["device"] = args.device
    results = model.train(**train_kwargs)

    save_dir = Path(getattr(results, "save_dir", None) or model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        fail(f"Training finished but {best} was not produced. The Android model was NOT changed.")

    print("\nValidating best.pt ...")
    metrics = YOLO(str(best)).val(data=str(resolved_yaml), imgsz=args.imgsz, project=str(RUNS_DIR),
                                  name=f"{args.name}_val")
    print(f"Validation mAP50-95: {metrics.box.map:.3f}  mAP50: {metrics.box.map50:.3f}")
    print(f"Best checkpoint: {best}")

    if args.export_to_app:
        from export_model import export_and_install
        export_and_install(best, imgsz=args.imgsz, install=True)
    else:
        print("Next: python export_model.py --weights", best, "--install")


if __name__ == "__main__":
    main()
