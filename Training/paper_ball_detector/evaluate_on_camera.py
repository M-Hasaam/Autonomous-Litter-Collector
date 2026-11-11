"""Score a paper-ball model on the two held-out test sets inside dataset_fixed/.

  test1  dataset_fixed/test/   25 pictures of the same kind as the training data (internet and phone photos).
                               Tells you whether the model learned the task.
  test2  dataset_fixed/test2/  23 pictures captured with the real ESP32-CAM, labelled by hand.
                               Tells you whether it works on YOUR camera.

Neither set is used for training. Read the two scores together: test1 good + test2 bad means the model does not
transfer to your camera (add your own labelled captures to training); test1 bad means the training itself is wrong.

Run:  python evaluate_on_camera.py [--model roboflow-paperball.pt] [--set test1|test2|both] [--imgsz 640] [--conf 0.25]
Needs: ultralytics (use the .pt model, not the .onnx).
Output: precision, recall, mAP50, mAP50-95 per set, plus annotated predictions in runs/test1/ and runs/test2/.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXED = HERE / "dataset_fixed"
SETS = {
    "test1": (FIXED / "data.yaml", "test", FIXED / "test" / "images"),
    "test2": (FIXED / "test2" / "test2.yaml", "val", FIXED / "test2" / "images"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(HERE / "roboflow-paperball.pt"))
    ap.add_argument("--set", choices=["test1", "test2", "both"], default="both")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold for the saved predictions")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        sys.exit(f"Model not found: {model_path}\nTrain it first (train_roboflow.py or train_paperball_colab.ipynb).")
    names = ["test1", "test2"] if args.set == "both" else [args.set]
    for name in names:
        yaml_path, split, images = SETS[name]
        if not images.is_dir():
            sys.exit(f"{name}: images not found: {images} (run fix_dataset.py)")

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    results = {}
    for name in names:
        yaml_path, split, images = SETS[name]
        n = len(list(images.glob("*.jpg")))
        print(f"\nEvaluating {model_path.name} on {name} ({n} pictures) ...")
        m = model.val(data=str(yaml_path), imgsz=args.imgsz, device=args.device, split=split, plots=False,
                      project=str(HERE / "runs"), name=f"{name}_val", exist_ok=True, workers=0)
        model.predict(source=str(images), imgsz=args.imgsz, conf=args.conf, device=args.device, save=True,
                      project=str(HERE / "runs"), name=name, exist_ok=True, verbose=False)
        results[name] = (n, m)

    print("\n=== Results ===")
    for name, (n, m) in results.items():
        print(f"{name} ({n:2d} pictures)  precision {m.box.mp:.3f}   recall {m.box.mr:.3f}   "
              f"mAP50 {m.box.map50:.3f}   mAP50-95 {m.box.map:.3f}")
    print(f"Annotated predictions: {HERE / 'runs'}")


if __name__ == "__main__":
    main()
