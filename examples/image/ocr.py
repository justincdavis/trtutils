# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""File showcasing the PP-OCR text detection + recognition pipeline."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from trtutils import set_log_level
from trtutils.models import PPOCR, PPOCRDetector, PPOCRRecognizer


def main() -> None:
    tmp_dir = Path(tempfile.gettempdir())
    det_onnx, det_engine = tmp_dir / "ppocrv4_det.onnx", tmp_dir / "ppocrv4_det.engine"
    rec_onnx, rec_engine = tmp_dir / "ppocrv4_rec.onnx", tmp_dir / "ppocrv4_rec.engine"

    if not det_onnx.exists():
        PPOCRDetector.download("ppocrv4_det", det_onnx)
    if not rec_onnx.exists():
        PPOCRRecognizer.download("ppocrv4_rec", rec_onnx)
    # both engines get dynamic input shapes: any image size / text line width
    if not det_engine.exists():
        PPOCRDetector.build(det_onnx, det_engine)
    if not rec_engine.exists():
        PPOCRRecognizer.build(rec_onnx, rec_engine)

    image = np.full((240, 900, 3), 255, np.uint8)
    for i, text in enumerate(["Hello from TensorRT", "PP-OCR text recognition", "12,500 / 3.14"]):
        cv2.putText(image, text, (30, 70 + 70 * i), cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 0, 0), 2)

    ocr = PPOCR(det_engine, rec_engine, warmup=True)

    t0 = time.perf_counter()
    results = ocr(image)
    t1 = time.perf_counter()
    print(f"Inference time: {round((t1 - t0) * 1000.0, 2)} ms")

    for box, text, score in results:
        print(f"{text!r} ({score:.3f}) at {box.astype(int).tolist()}")

    # the stages can also be used on their own
    boxes, _ = ocr.detector(image)
    print(f"Detected {len(boxes)} text boxes")
    print(ocr.recognizer(image[40:100, 20:500]))


if __name__ == "__main__":
    set_log_level("ERROR")
    main()
