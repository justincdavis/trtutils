# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Demo showcasing YOLO inference on a webcam."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from statistics import mean

import cv2

from trtutils import build_engine
from trtutils.image import draw_bboxes, draw_text
from trtutils.models import YOLO

_ONNX = Path(__file__).parent / "data" / "yolov10n.onnx"
_ENGINE = Path(__file__).parent / "data" / "yolov10n.engine"
_FPS_BUFFER_SIZE = 30


def _main(source: str) -> None:
    if not _ENGINE.exists():
        build_engine(
            _ONNX,
            _ENGINE,
            fp16=True,
        )

    fps_buffer = [1.0] * _FPS_BUFFER_SIZE
    yolo = YOLO(
        _ENGINE,
        warmup=True,
        warmup_iterations=10,
        preprocessor="trt",
        resize_method="letterbox",
    )

    cap = cv2.VideoCapture(source)
    fid = -1
    while True:
        got, frame = cap.read()
        if not got:
            break
        fid += 1

        t0 = time.time()
        dets = yolo.end2end([frame])[0]
        t1 = time.time()
        t_ms = (t1 - t0) * 1000.0
        fps = 1000.0 / t_ms
        fps_buffer[fid % _FPS_BUFFER_SIZE] = fps
        avg_fps = mean(fps_buffer)

        canvas = draw_bboxes(frame, [bbox for bbox, _, _ in dets])
        draw_text(canvas, f"{avg_fps:.2f} FPS", (10, 30))
        cv2.imshow("YOLO Demo", canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    del yolo
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("YOLO Demo")
    parser.add_argument("--source", type=int, default=0)
    args = parser.parse_args()

    _main(args.source)
