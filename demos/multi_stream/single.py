# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Demo showcasing using a single stream to buffer video frames then perform inference."""

from __future__ import annotations

import time
from pathlib import Path

import cv2ext
import trtutils


VIDEO_FILES = [
    Path("videos/mot17_02.mp4"),
    Path("videos/mot17_04.mp4"),
    Path("videos/mot17_05.mp4"),
    Path("videos/mot17_09.mp4"),
    Path("videos/mot17_10.mp4"),
    Path("videos/mot17_11.mp4"),
    Path("videos/mot17_13.mp4"),
]

ENGINE_FILE = Path("engines/yoloxm_gpu.engine")


def _main() -> None:
    yolo = trtutils.impls.yolo.YOLO(ENGINE_FILE, warmup_iterations=10, warmup=True)

    for video_file in VIDEO_FILES:
        with cv2ext.Display(video_file.stem) as display:
            for i, frame in cv2ext.IterableVideo(video_file):
                t0 = time.time()
                dets = yolo.end2end(frame)
                t1 = time.time()

                canvas = cv2ext.detection.draw_detections(frame, dets)
                cv2ext.image.draw.text(canvas, f"{i} - FPS: {1 / (t1 - t0):.2f}", (10, 30), color=(0, 0, 255))
                display.update(canvas)


if __name__ == "__main__":
    _main()
