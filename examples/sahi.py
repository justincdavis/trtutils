# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""File showcasing the Detector class."""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from trtutils import set_log_level
from trtutils.image import SAHI, Detector


def main() -> None:
    """Run the example."""
    engine_dir = Path(__file__).parent.parent / "data" / "engines"
    engine_path = engine_dir / "trt_yolov10n.engine"

    img = cv2.imread(str(Path(__file__).parent.parent / "data" / "street.jpg"))

    detector = Detector(engine_path, warmup=True, preprocessor="trt")

    t0 = time.perf_counter()
    bboxes = detector.end2end(img)
    t1 = time.perf_counter()
    standalone_time = t1 - t0

    sahi = SAHI(detector)
    t0 = time.perf_counter()
    sahi_bboxes = sahi.end2end(img)
    t1 = time.perf_counter()
    sahi_time = t1 - t0

    print(
        f"Detector: bboxes: {len(bboxes)}, in {round((standalone_time) * 1000.0, 2)} ms"
    )
    print(
        f"SAHI:     bboxes: {len(sahi_bboxes)}, in {round((sahi_time) * 1000.0, 2)} ms"
    )

    del detector


if __name__ == "__main__":
    set_log_level("ERROR")
    main()
