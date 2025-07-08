# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""File showcasing the Detector class."""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from trtutils import set_log_level
from trtutils.image import Detector


def main() -> None:
    """Run the example."""
    engine_dir = Path(__file__).parent.parent / "data" / "engines"
    engines = [
        engine_dir / "trt_yolov7t.engine",
        engine_dir / "trt_yolov8n.engine",
        engine_dir / "trt_yolov9t.engine",
        engine_dir / "trt_yolov10n.engine",
        engine_dir / "trt_yolov7t_dla.engine",
        engine_dir / "trt_yolov8n_dla.engine",
        engine_dir / "trt_yolov9t_dla.engine",
        engine_dir / "trt_yolov10n_dla.engine",
    ]

    img = cv2.imread(str(Path(__file__).parent.parent / "data" / "horse.jpg"))

    for engine in engines:
        detector = Detector(engine, warmup=True, preprocessor="cuda")
        print(detector.name)

        t0 = time.perf_counter()
        output = detector.run(img)
        bboxes = detector.get_detections(output)
        t1 = time.perf_counter()

        print(f"RUN, bboxes: {bboxes}, in {round((t1 - t0) * 1000.0, 2)}")

        # OR

        # end2end makes a few memory optimzations by avoiding extra GPU
        # memory transfers
        t0 = time.perf_counter()
        bboxes = detector.end2end(img)
        t1 = time.perf_counter()

        print(f"END2END: bboxes: {bboxes}, in {round((t1 - t0) * 1000.0, 2)}")

        del detector


if __name__ == "__main__":
    set_log_level("ERROR")
    main()
