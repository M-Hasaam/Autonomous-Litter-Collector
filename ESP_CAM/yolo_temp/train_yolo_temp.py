"""Fine-tune YOLO11n on the temporary paper-ball dataset and save it as yolo-temp.pt / yolo-temp.onnx.

This is a throw-away model for testing distance estimation (camera_distance.py). It does NOT touch the app's model
(app/src/main/assets/litter_detector.tflite) and does not use training/export_model.py.

Needs the Ultralytics environment (see training/requirements.txt, at least: ultralytics torch onnx).
Run:  python train_yolo_temp.py [--epochs 80]
"""
import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_WEIGHTS = HERE.parent.parent / "training" / "yolo11n.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(BASE_WEIGHTS))
    results = model.train(
        data=str(HERE / "dataset" / "dataset.yaml"), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, project=str(HERE / "runs"), name="yolo-temp", exist_ok=True, workers=0,
        flipud=0.0, degrees=0.0, plots=False, patience=0,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    shutil.copy2(best, HERE / "yolo-temp.pt")
    onnx = Path(YOLO(str(HERE / "yolo-temp.pt")).export(format="onnx", imgsz=args.imgsz, simplify=True, opset=12))
    if onnx.resolve() != (HERE / "yolo-temp.onnx").resolve():  # the export normally lands right next to the .pt
        shutil.copy2(onnx, HERE / "yolo-temp.onnx")
    print(f"\nSaved {HERE / 'yolo-temp.pt'} and {HERE / 'yolo-temp.onnx'}")


if __name__ == "__main__":
    main()
