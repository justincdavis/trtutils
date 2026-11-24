# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""File showcasing the SemanticSegmenter class with torchvision DeepLabV3."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from trtutils import set_log_level
from trtutils.models import DeepLabV3

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# the 21 Pascal VOC classes of the default DeepLabV3 weights
VOC_CLASSES = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor",
)  # fmt: skip


def main() -> None:
    tmp_dir = Path(tempfile.gettempdir())
    onnx_path = tmp_dir / "deeplabv3_mobilenet_v3_large.onnx"
    engine_path = tmp_dir / "deeplabv3_mobilenet_v3_large.engine"

    if not onnx_path.exists():
        print("Downloading DeepLabV3 MobileNetV3-Large ONNX model...")
        DeepLabV3.download("deeplabv3_mobilenet_v3_large", onnx_path)

    # build() appends the argmax to the engine, so it outputs a class map instead of logits
    if not engine_path.exists():
        DeepLabV3.build(onnx_path, engine_path)

    image_path = DATA_DIR / "horse.jpg"
    image = cv2.imread(str(image_path))
    if image is None:
        msg = f"Could not read image: {image_path}"
        raise FileNotFoundError(msg)

    segmenter = DeepLabV3(engine_path, warmup=True)

    t0 = time.perf_counter()
    class_map = segmenter.end2end(image)
    t1 = time.perf_counter()
    print(f"Inference time: {round((t1 - t0) * 1000.0, 2)} ms")

    # class_map is (H, W) uint8 class ids at the image's resolution
    ids, counts = np.unique(class_map, return_counts=True)
    for class_id, count in zip(ids, counts):
        print(f"{VOC_CLASSES[class_id]}: {100.0 * count / class_map.size:.1f}% of pixels")

    # one color per class, blended over the image where something was found
    palette = np.random.default_rng(0).integers(0, 255, (len(VOC_CLASSES), 3), dtype=np.uint8)
    colors = palette[class_map]
    overlay = np.where(
        (class_map > 0)[..., None], cv2.addWeighted(image, 0.4, colors, 0.6, 0), image
    )

    output_path = DATA_DIR / "horse_semantic.jpg"
    cv2.imwrite(str(output_path), overlay)
    print(f"Saved class map overlay to {output_path}")


if __name__ == "__main__":
    set_log_level("ERROR")
    main()
