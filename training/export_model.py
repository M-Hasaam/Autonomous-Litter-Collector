"""Export a YOLO11n checkpoint to the TFLite + labels.txt format used by the Android app.

Examples:
    python export_model.py                                  # export pretrained yolo11n.pt, don't touch the app
    python export_model.py --install                        # ... and install into app/src/main/assets/
    python export_model.py --weights runs/litter_yolo11n/weights/best.pt --install

Output files (installed under app/src/main/assets/):
    litter_detector.tflite   float32, input [1, S, S, 3] RGB scaled 0..1, output [1, 4+nc, N]
    labels.txt               one class name per line, in class-index order

The app model is replaced only after export AND validation of the new files succeed; the
previous model/labels are copied to *.bak first and restored if the install fails.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP_ASSETS = HERE.parent / "app" / "src" / "main" / "assets"
MODEL_NAME = "litter_detector.tflite"
LABELS_NAME = "labels.txt"


def _load_interpreter(model_path: Path):
    """Return a TFLite interpreter from whichever runtime is installed."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter
        except ImportError as exc:
            raise RuntimeError("Install tensorflow (see requirements.txt) to validate the model.") from exc
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def validate_export(model_path: Path, labels: list[str], imgsz: int) -> None:
    """Check the exported file really has the layout the Android decoder expects."""
    interpreter = _load_interpreter(model_path)
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()
    in_shape = [int(v) for v in inp["shape"]]
    if in_shape != [1, imgsz, imgsz, 3]:
        raise RuntimeError(f"Unexpected input shape {in_shape}, expected [1, {imgsz}, {imgsz}, 3].")
    if inp["dtype"].__name__ != "float32":
        raise RuntimeError(f"Expected float32 input, got {inp['dtype']}.")
    if len(out) != 1:
        raise RuntimeError(f"Expected 1 output tensor, got {len(out)} (was NMS baked in?).")
    out_shape = [int(v) for v in out[0]["shape"]]
    if len(out_shape) != 3 or out_shape[0] != 1 or (4 + len(labels)) not in out_shape[1:]:
        raise RuntimeError(
            f"Output shape {out_shape} does not contain 4 + {len(labels)} classes = {4 + len(labels)}."
        )
    print(f"Validated: input {in_shape}, output {out_shape}, {len(labels)} classes.")


def export_and_install(weights: Path, imgsz: int = 640, install: bool = False,
                       assets_dir: Path = APP_ASSETS, out_dir: Path | None = None) -> Path:
    """Export `weights` and (optionally) install into the Android assets. Returns the .tflite path."""
    from ultralytics import YOLO

    weights = Path(weights)
    if weights.suffix == ".pt" and weights.name != "yolo11n.pt" and not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")

    model = YOLO(str(weights))  # also downloads yolo11n.pt if needed
    names = model.names
    labels = [names[i] for i in sorted(names)]

    # Ultralytics' own `format="tflite"` path is unavailable on Windows with Ultralytics >= 8.4
    # and breaks across onnx2tf versions, so go checkpoint -> static ONNX -> onnx2tf -> float32 TFLite.
    work = Path(tempfile.mkdtemp(prefix="litter_build_"))
    onnx_dir = work / "onnx"
    onnx_dir.mkdir()
    # Export from a copy so the .onnx files land in the temp folder, not next to your checkpoint.
    ckpt_copy = onnx_dir / "model.pt"
    shutil.copyfile(model.ckpt_path, ckpt_copy)
    onnx_model = YOLO(str(ckpt_copy))
    onnx_path = Path(onnx_model.export(format="onnx", imgsz=imgsz, opset=12, simplify=True,
                                       dynamic=False, half=False))
    tf_dir = work / "tf"
    subprocess.run(
        [sys.executable, "-m", "onnx2tf", "-i", str(onnx_path), "-o", str(tf_dir),
         "-ois", f"images:1,3,{imgsz},{imgsz}", "-n"],
        check=True, cwd=work,  # cwd=work keeps onnx2tf's scratch files out of the repo
    )
    exported = tf_dir / f"{onnx_path.stem}_float32.tflite"
    if not exported.is_file():
        raise RuntimeError(f"onnx2tf did not produce {exported.name}")

    validate_export(exported, labels, imgsz)

    # Stage both files, then swap them in so a failure never leaves a half-updated app model.
    stage = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="litter_export_"))
    stage.mkdir(parents=True, exist_ok=True)
    staged_model = stage / MODEL_NAME
    staged_labels = stage / LABELS_NAME
    shutil.copyfile(exported, staged_model)
    staged_labels.write_text("\n".join(labels) + "\n", encoding="utf-8")
    print(f"Exported model + labels to {stage}")

    if not install:
        print("Not installed into the app (use --install).")
        return staged_model

    assets_dir.mkdir(parents=True, exist_ok=True)
    targets = {assets_dir / MODEL_NAME: staged_model, assets_dir / LABELS_NAME: staged_labels}
    backups = {}
    try:
        for target, src in targets.items():
            tmp = target.with_name(target.name + ".new")
            shutil.copyfile(src, tmp)
            if target.exists():
                backups[target] = target.with_name(target.name + ".bak")
                shutil.copyfile(target, backups[target])
            os.replace(tmp, target)
    except Exception:
        for target, backup in backups.items():
            shutil.copyfile(backup, target)
        raise
    finally:
        for target in targets:
            target.with_name(target.name + ".new").unlink(missing_ok=True)
    for backup in backups.values():
        backup.unlink(missing_ok=True)
    print(f"Installed into {assets_dir}. Rebuild/run the app to use it.")
    return assets_dir / MODEL_NAME


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--weights", default="yolo11n.pt",
                        help="Checkpoint to export (default: pretrained yolo11n.pt, downloaded if needed).")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Model input size. 640 is standard; 320 is ~4x faster on weak phones.")
    parser.add_argument("--install", action="store_true",
                        help=f"Copy the result to app/src/main/assets/{MODEL_NAME} and {LABELS_NAME}.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Keep the exported files in this folder.")
    args = parser.parse_args()
    try:
        export_and_install(Path(args.weights), imgsz=args.imgsz, install=args.install, out_dir=args.out_dir)
    except Exception as exc:  # keep the previous app model on any failure
        print(f"ERROR: export failed, the Android app model was NOT changed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
