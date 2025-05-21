# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from pathlib import Path

import cv2ext
import trtutils


_ONNX = Path(__file__).parent / "data" / "yolov10n.onnx"
_ENGINE = Path(__file__).parent / "data" / "yolov10n.engine"

def main():
    yolo = trtutils.impls.yolo.YOLO(

    )


    for fid, frame in cv2ext.IterableVideo(0):
        dets = yolo.end2end(frame)


if __name__ == "__main__":
    main()
