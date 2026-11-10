# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""File showcasing the DeiT classifier at a custom input size."""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from trtutils import set_log_level
from trtutils.models import DeiT

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def main() -> None:
    # any size divisible by 16 works, the position embeddings are resampled
    imgsz = 256
    onnx_path = DATA_DIR / "deit" / f"deit_small_distilled_{imgsz}.onnx"
    engine_path = DATA_DIR / "deit" / f"deit_small_distilled_{imgsz}.engine"

    if not onnx_path.exists():
        print("Downloading DeiT ONNX model...")
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        DeiT.download("deit_small_distilled", onnx_path, imgsz=imgsz)

    if not engine_path.exists():
        DeiT.build(onnx_path, engine_path, imgsz=imgsz)

    image_path = DATA_DIR / "horse.jpg"
    image = cv2.imread(str(image_path))
    if image is None:
        msg = f"Could not read image: {image_path}"
        raise FileNotFoundError(msg)

    # center crop with crop_pct=0.9, the DeiT evaluation preprocessing
    model = DeiT(engine_path, warmup=True)

    t0 = time.perf_counter()
    classifications = model.end2end(image, top_k=5)
    t1 = time.perf_counter()
    print(f"Top-5 (class id, score): {classifications}")
    print(f"Inference time: {round((t1 - t0) * 1000.0, 2)} ms")

    del model


if __name__ == "__main__":
    set_log_level("ERROR")
    main()
