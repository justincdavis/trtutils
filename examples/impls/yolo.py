# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""File showcasing the TRTEngine class."""

from __future__ import annotations

from pathlib import Path

import cv2

from trtutils.impls.yolo import YOLO


# This example shows how to use the TRTEngine class
# on running a Yolo model with a single input image.
# The Yolo model is not included in this repository.
# This works with a yolov7 engine created by
# using the export script locating in the yolov7 repository.
# Then generate an engine using TensorRT by:
#  trtexec --onnx=yolo.onnx --saveEngine=yolo.engine
# The resulting engine can be used with this example.
def main() -> None:
    """Run the example."""
    yolo = YOLO(
        Path(__file__).parent.parent.parent / "data" / "engines" / "trt_yolov7t.engine",
        version=7,
        warmup=True,
    )
    img = cv2.imread(str(Path(__file__).parent.parent.parent / "data" / "horse.jpg"))

    output = yolo.run(img)

    bboxes = yolo.get_detections(output)

    print(bboxes)


if __name__ == "__main__":
    main()
