# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""File showcasing Detector accepting a GPU-resident image via core.Buffer."""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from trtutils import set_log_level
from trtutils.core import Buffer, MemoryLocation
from trtutils.image import Detector


def main() -> None:
    engine_path = (
        Path(__file__).resolve().parent.parent.parent / "data" / "engines" / "trt_yolov10n.engine"
    )

    img_path = str(Path(__file__).resolve().parent.parent.parent / "data" / "horse.jpg")
    img = cv2.imread(img_path)
    if img is None:
        err_msg = f"Failed to load image from {img_path}"
        raise FileNotFoundError(err_msg)

    detector = Detector(engine_path, warmup=True, preprocessor="cuda", cuda_graph=True)

    # HWC uint8 host array -> device Buffer. In a real GPU pipeline this
    # buffer would already live on the device, e.g. wrapped from a torch
    # tensor or cupy array instead of copied up from the host here:
    #   buf = Buffer.from_cuda_array(torch_tensor_or_cupy_array)
    buf = Buffer.from_array(img, MemoryLocation.DEVICE)

    # the cuda/trt preprocessors consume a device Buffer in place, with no
    # host-to-device copy; the cpu preprocessor would copy it to host once
    t0 = time.perf_counter()
    bboxes = detector.end2end(buf)
    t1 = time.perf_counter()
    print(f"END2END (device Buffer): bboxes: {len(bboxes)}, in {round((t1 - t0) * 1000.0, 2)} ms")

    buf.free()
    del detector


if __name__ == "__main__":
    set_log_level("ERROR")
    main()
