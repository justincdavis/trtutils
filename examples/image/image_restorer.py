# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""File showcasing the ImageRestorer class with Real-ESRGAN super-resolution."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2

from trtutils import set_log_level
from trtutils.models import RealESRGAN

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def main() -> None:
    image_path = DATA_DIR / "horse.jpg"
    image = cv2.imread(str(image_path))
    if image is None:
        msg = f"Could not read image: {image_path}"
        raise FileNotFoundError(msg)

    # the engine input is static and images are never rescaled: build at the image size
    height, width = image.shape[:2]
    tmp_dir = Path(tempfile.gettempdir())
    onnx_path = tmp_dir / f"realesrgan_general_x4v3_{height}x{width}.onnx"
    engine_path = tmp_dir / f"realesrgan_general_x4v3_{height}x{width}.engine"

    if not onnx_path.exists():
        print("Downloading Real-ESRGAN general-x4v3 ONNX model...")
        RealESRGAN.download("realesrgan_general_x4v3", onnx_path, imgsz=(height, width))

    if not engine_path.exists():
        RealESRGAN.build(onnx_path, engine_path)

    restorer = RealESRGAN(engine_path, warmup=True)

    t0 = time.perf_counter()
    restored = restorer.end2end(image)
    t1 = time.perf_counter()
    print(f"Inference time: {round((t1 - t0) * 1000.0, 2)} ms")

    # restored is a uint8 BGR image, scale times the input size
    print(f"{image.shape[:2]} -> {restored.shape[:2]} (scale {restorer.scale})")
    output_path = DATA_DIR / "horse_x4.png"
    cv2.imwrite(str(output_path), restored)
    print(f"Saved restored image to {output_path}")


if __name__ == "__main__":
    set_log_level("ERROR")
    main()
