# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import cv2ext
import numpy as np
import trtutils
from trtutils.impls.yolo import YOLO


def main() -> None:
    trtutils.set_log_level("DEBUG")

    parser = argparse.ArgumentParser("Evaluate a YOLO engine.")
    parser.add_argument(
        "--engine",
        required=True,
        type=Path,
        help="The path to the yolo engine.",
    )
    parser.add_argument(
        "--version",
        required=True,
        type=int,
        help="The version of YOLO the engine is.",
    )
    args = parser.parse_args()

    img = cv2.imread(str(Path(__file__).parent.parent / "data" / "people.jpeg"))

    yolo = YOLO(args.engine, args.version)

    output = yolo(img, preprocessed=False, postprocess=True)

    print(output)

    detections = yolo.get_detections()
    for det in detections:
        print(det)

    bboxes = [bbox for (bbox, _, _) in detections]

    canvas = cv2ext.bboxes.draw_bboxes(img, bboxes)
    canvas = cv2.resize(canvas, None, fx=3.0, fy=3.0)

    cv2.imshow("Detections", canvas)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
