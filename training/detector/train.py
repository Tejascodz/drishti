"""Train a YOLO26-pose plate detector (box + 4 corners) and export it for Drishti.

    pip install "ultralytics>=8.4.0"
    python training/detector/train.py --data training/detector/plates_pose.yaml --model yolo26s-pose.pt
    python training/detector/train.py --export runs/pose/plates/weights/best.pt --imgsz 960

YOLO26 is NMS-free end-to-end, so the exported ONNX feeds Drishti directly
(output (N, 300, 6 + 12) is auto-detected). Notes for plates:
  * imgsz 960-1280: plates are small; resolution matters more than model size.
  * Two-stage recipe: pre-train corners on CCPD (convert.py ccpd), then fine-tune
    on Indian data mixed with ~20% CCPD to keep the corner head sharp.
  * Ultralytics is AGPL-3.0; use its enterprise licence for closed deployments, or
    keep the MIT open-image-models detector (Drishti's default).
"""

from __future__ import annotations

import argparse


def train(a) -> None:
    from ultralytics import YOLO

    model = YOLO(a.model)
    model.train(
        data=a.data,
        imgsz=a.imgsz,
        epochs=a.epochs,
        batch=a.batch,
        device=a.device,
        workers=a.workers,
        project=a.project,
        name=a.name,
        # geometry: plates are rigid planes seen at moderate angles
        degrees=5.0,
        shear=2.0,
        perspective=0.0005,
        translate=0.1,
        scale=0.6,
        fliplr=0.5,  # corners stay consistent through flip_idx
        mosaic=1.0,
        close_mosaic=15,
        # photometric: day / night / sodium-lamp colour casts
        hsv_h=0.02,
        hsv_s=0.6,
        hsv_v=0.5,
        cos_lr=True,
        patience=40,
        plots=True,
    )
    print("best weights:", model.trainer.best)


def export(a) -> None:
    from ultralytics import YOLO

    m = YOLO(a.export)
    onnx = m.export(format="onnx", imgsz=a.imgsz, dynamic=True, simplify=True, nms=False)
    print("ONNX (end-to-end, NMS-free):", onnx)
    if a.coreml:
        print("CoreML:", m.export(format="coreml", imgsz=a.imgsz, nms=False))
    if a.tensorrt:
        print("TensorRT:", m.export(format="engine", imgsz=a.imgsz, half=True, dynamic=False))
    print("use it with:  models: {plate_detector: <path to .onnx>}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="training/detector/plates_pose.yaml")
    ap.add_argument("--model", default="yolo26s-pose.pt", help="yolo26n/s/m-pose.pt (n for Jetson / CPU)")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--project", default="runs/pose")
    ap.add_argument("--name", default="plates")
    ap.add_argument("--export", help="path to trained .pt; export instead of training")
    ap.add_argument("--coreml", action="store_true")
    ap.add_argument("--tensorrt", action="store_true")
    a = ap.parse_args()
    export(a) if a.export else train(a)


if __name__ == "__main__":
    main()
