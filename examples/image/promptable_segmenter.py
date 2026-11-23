# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""File showcasing MobileSAM promptable segmentation from points, boxes, and detections."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from trtutils import set_log_level
from trtutils.models import MobileSAM, MobileSAMDecoder, MobileSAMEncoder, YOLOv8


def _engine(cls: type[MobileSAMEncoder | MobileSAMDecoder | YOLOv8], model: str) -> Path:
    """Download and build one engine into the temp directory, once."""
    tmp_dir = Path(tempfile.gettempdir())
    onnx, engine = tmp_dir / f"{model}.onnx", tmp_dir / f"{model}.engine"
    if not onnx.exists():
        cls.download(model, onnx)
    if not engine.exists():
        cls.build(onnx, engine)
    return engine


def main() -> None:
    # the image encoder runs once per image, the small decoder once per prompt; the larger
    # SAM models work the same way: SAM, SAMEncoder and SAMDecoder on "sam_vit_b_*", and
    # SAM2, SAM2Encoder and SAM2Decoder on "sam2_1_hiera_tiny_*"
    sam = MobileSAM(
        _engine(MobileSAMEncoder, "mobile_sam_encoder"),
        _engine(MobileSAMDecoder, "mobile_sam_decoder"),
        warmup=True,
    )
    image = cv2.imread(str(Path(__file__).parents[2] / "data" / "horse.jpg"))
    if image is None:
        err_msg = "data/horse.jpg not found"
        raise FileNotFoundError(err_msg)

    t0 = time.perf_counter()
    sam.set_image(image)
    t1 = time.perf_counter()
    # prompts are in image pixels: a foreground point on the horse ...
    ((point_mask, point_iou),) = sam.predict(points=np.array([[230, 110]]))
    # ... or a box around it, which is not ambiguous, so take the single mask
    # (the first call at a new prompt shape also sets up its execution context)
    t2 = time.perf_counter()
    ((box_mask, box_iou),) = sam.predict(boxes=np.array([65, 45, 392, 272]), multimask=False)
    t3 = time.perf_counter()
    print(f"set_image {(t1 - t0) * 1000:.2f} ms, predict {(t3 - t2) * 1000:.2f} ms")
    print(f"point: {point_mask.sum()} px (IoU {point_iou:.3f})")
    print(f"box: {box_mask.sum()} px (IoU {box_iou:.3f})")

    # detector boxes -> one mask per detection
    detector = YOLOv8(_engine(YOLOv8, "yolov8n"), warmup=True)
    detections = detector.end2end(image)
    boxes = np.array([bbox for bbox, _, _ in detections], np.float32).reshape(-1, 4)
    masks = sam.end2end(image, boxes=boxes, multimask=False)
    for (bbox, score, class_id), (mask, _) in zip(detections, masks):
        print(f"class {class_id} ({score:.2f}) at {bbox}: mask of {mask.sum()} px")

    overlay = image.copy()
    overlay[box_mask.astype(bool)] = (0, 0, 255)
    cv2.imwrite(str(Path(tempfile.gettempdir()) / "horse_mask.jpg"), overlay)


if __name__ == "__main__":
    set_log_level("ERROR")
    main()
