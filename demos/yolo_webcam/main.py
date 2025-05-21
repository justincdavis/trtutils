# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import argparse
from pathlib import Path

import cv2ext
import trtutils


_ONNX = Path(__file__).parent / "data" / "yolov10n.onnx"
_ENGINE = Path(__file__).parent / "data" / "yolov10n.engine"

def main(source: str):
    if not _ENGINE.exists():
        trtutils.build_engine(
            _ONNX,
            _ENGINE,
            fp16=True,
        )

    yolo = trtutils.impls.yolo.YOLO(
        _ENGINE,
        warmup=True,
        warmup_iterations=10,
        preprocessor="trt",
        resize_method="letterbox",
    )

    display = cv2ext.Display("YOLO Demo")
    for fid, frame in cv2ext.IterableVideo(source):
        if display.stopped:
            break

        dets = yolo.end2end(frame)
        canvas = cv2ext.bboxes.draw_bboxes(frame, [bbox for bbox, _, _ in dets])
        cv2ext.image.draw.text(canvas, f"{fid}", (10, 30))
        display.update(canvas)

    del yolo
    display.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("YOLO Demo")
    parser.add_argument("--source", type=str, default=0)
    args = parser.parse_args()

    main(args.source)
