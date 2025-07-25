# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""File showcasing the Detector class."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import cv2ext

from trtutils import set_log_level
from trtutils.image import SAHI, Detector


def main() -> None:
    parser = argparse.ArgumentParser(description="SAHI example.")
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the detections using cv2ext.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output.",
    )
    args = parser.parse_args()

    engine_dir = Path(__file__).parent.parent / "data" / "engines"
    engine_path = engine_dir / "trt_yolov10n.engine"

    img_path = str(Path(__file__).parent.parent / "data" / "cars.jpeg")
    img = cv2.imread(img_path)
    if img is None:
        err_msg = f"Failed to load image from {img_path}"
        raise FileNotFoundError(err_msg)

    detector = Detector(engine_path, warmup=True, preprocessor="trt", verbose=args.verbose)

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

    if args.display:
        canvas = cv2ext.bboxes.draw_bboxes(
            img.copy(), [bbox for bbox, _, _ in sahi_bboxes], color=(0, 255, 0)
        )
        canvas = cv2ext.bboxes.draw_bboxes(
            canvas, [bbox for bbox, _, _ in bboxes], color=(0, 0, 255)
        )
        cv2.imshow("SAHI Detections", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    del detector


if __name__ == "__main__":
    set_log_level("ERROR")
    main()
