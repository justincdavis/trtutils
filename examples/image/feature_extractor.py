# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""File showcasing the FeatureExtractor class with DINOv3."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from trtutils import set_log_level
from trtutils.models import DINOv3

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def main() -> None:
    tmp_dir = Path(tempfile.gettempdir())
    onnx_path = tmp_dir / "dinov3_vits16_256.onnx"
    engine_path = tmp_dir / "dinov3_vits16_256.engine"

    if not onnx_path.exists():
        print("Downloading DINOv3 ViT-S/16 ONNX model...")
        DINOv3.download("dinov3_vits16", onnx_path, imgsz=256)

    if not engine_path.exists():
        # fp16, with the attention matmul and softmax kept in fp32
        DINOv3.build(onnx_path, engine_path)

    images = {}
    for name in ("horse.jpg", "people.jpeg", "street.jpg"):
        image = cv2.imread(str(DATA_DIR / name))
        if image is None:
            msg = f"Could not read image: {DATA_DIR / name}"
            raise FileNotFoundError(msg)
        images[name] = image

    # normalize=True: unit-length embeddings, so a dot product is a cosine similarity
    extractor = DINOv3(engine_path, warmup=True, normalize=True)

    t0 = time.perf_counter()
    embeddings = {name: extractor.end2end(image) for name, image in images.items()}
    t1 = time.perf_counter()
    print(f"Embedded {len(images)} images in {round((t1 - t0) * 1000.0, 2)} ms")
    print(f"Embedding shape: {embeddings['horse.jpg'].shape}")

    names = list(embeddings)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            similarity = float(np.dot(embeddings[first], embeddings[second]))
            print(f"cosine({first}, {second}) = {similarity:.3f}")

    # the dense (C, h, w) patch features, e.g. for retrieval or a light segmentation head
    outputs = extractor.run(images["horse.jpg"])
    feature_map = extractor.get_feature_maps(outputs)
    print(f"Feature map shape: {feature_map.shape}")

    # one embedding per box, e.g. the detections of a frame for tracking or re-ID
    boxes = [(0, 0, 400, 300), (500, 200, 900, 700), (1200, 400, 1600, 1000)]
    box_embeddings = extractor.end2end_boxes(images["street.jpg"], boxes)
    print(f"Box embeddings shape: {box_embeddings.shape}")


if __name__ == "__main__":
    set_log_level("ERROR")
    main()
