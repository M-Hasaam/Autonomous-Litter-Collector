"""Run a checkpoint on an image or a folder of images and save annotated copies.

Example:
    python predict.py --weights runs/litter_yolo11n/weights/best.pt --source photos/
    python predict.py --source some_image.jpg          # pretrained yolo11n.pt, general COCO classes

Annotated images are written to training/predictions/<name>/.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREDICTIONS_DIR = HERE / "predictions"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--weights", default="yolo11n.pt", help="Checkpoint (.pt). Default: pretrained yolo11n.pt.")
    parser.add_argument("--source", required=True, type=Path, help="Image file or folder of images.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", default="predict")
    args = parser.parse_args()

    if not args.source.exists():
        print(f"ERROR: source not found: {args.source}", file=sys.stderr)
        sys.exit(1)
    weights = Path(args.weights)
    if weights.suffix == ".pt" and weights.name != "yolo11n.pt" and not weights.is_file():
        print(f"ERROR: weights not found: {weights}", file=sys.stderr)
        sys.exit(1)

    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.predict(
        source=str(args.source), conf=args.conf, imgsz=args.imgsz, save=True,
        project=str(PREDICTIONS_DIR), name=args.name, exist_ok=True,
    )
    print(f"Annotated results saved under {PREDICTIONS_DIR / args.name}")


if __name__ == "__main__":
    main()
