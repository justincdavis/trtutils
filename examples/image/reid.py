# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""File showcasing person re-identification: YOLOv10 finds people, OSNet embeds each one."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from trtutils import set_log_level
from trtutils.models import OSNet, YOLOv10

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_PERSON = 0  # COCO class id


def main() -> None:
    tmp_dir = Path(tempfile.gettempdir())
    det_onnx = tmp_dir / "yolov10n_640.onnx"
    det_engine = tmp_dir / "yolov10n_640.engine"
    reid_onnx = tmp_dir / "osnet_x0_25_msmt17_256x128.onnx"
    reid_engine = tmp_dir / "osnet_x0_25_msmt17_256x128_b32.engine"

    if not det_onnx.exists():
        print("Downloading YOLOv10n ONNX model...")
        YOLOv10.download("yolov10n", det_onnx)
    if not det_engine.exists():
        YOLOv10.build(det_onnx, det_engine)
    if not reid_onnx.exists():
        print("Downloading OSNet x0.25 (MSMT17) ONNX model...")
        OSNet.download("osnet_x0_25_msmt17", reid_onnx)
    if not reid_engine.exists():
        # a dynamic batch: one engine call embeds 1 to 32 people
        OSNet.build(reid_onnx, reid_engine, batch_size=32)

    image = cv2.imread(str(DATA_DIR / "people.jpeg"))
    if image is None:
        msg = f"Could not read image: {DATA_DIR / 'people.jpeg'}"
        raise FileNotFoundError(msg)

    detector = YOLOv10(det_engine, warmup=True)
    # embeddings are unit length by default, so a dot product is a cosine similarity
    reid = OSNet(reid_engine, warmup=True)

    detections = detector.end2end(image, conf_thres=0.5)
    boxes = [bbox for bbox, _score, class_id in detections if class_id == _PERSON]
    print(f"Found {len(boxes)} people: {boxes}")

    t0 = time.perf_counter()
    embeddings = reid.end2end_boxes(image, boxes)
    t1 = time.perf_counter()
    print(f"Embedded {len(boxes)} people in {round((t1 - t0) * 1000.0, 2)} ms")
    print(f"Embeddings shape: {embeddings.shape}")

    # a tracker matches detections across frames by this similarity (or 1 - it, a distance)
    similarity = embeddings @ embeddings.T
    print("Pairwise cosine similarity:")
    print(np.array2string(similarity, precision=3, suppress_small=True))


if __name__ == "__main__":
    set_log_level("ERROR")
    main()
