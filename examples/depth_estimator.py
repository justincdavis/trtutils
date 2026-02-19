# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""File showcasing the DepthEstimator class."""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from trtutils import set_log_level
from trtutils.builder import build_engine
from trtutils.download import download
from trtutils.image import DepthEstimator


def main() -> None:
    onnx_path = Path("/tmp/depth_anything_v2_small.onnx")  # noqa: S108
    engine_path = Path("/tmp/depth_anything_v2_small.engine")  # noqa: S108

    if not onnx_path.exists():
        print("Downloading DepthAnythingV2 small ONNX model...")
        download("depth_anything_v2_small", onnx_path, imgsz=518, simplify=True, accept=True)

    if not engine_path.exists():
        build_engine(onnx_path, engine_path, fp16=True, shapes=[("input", (1, 3, 518, 518))])

    depth_estimator = DepthEstimator(engine_path, warmup=True, preprocessor="cuda", cuda_graph=False)
    webcam = cv2.VideoCapture(0)

    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        t0 = time.perf_counter()
        depth_maps = depth_estimator.end2end([frame])
        t1 = time.perf_counter()
        print(f"Time: {round((t1 - t0) * 1000.0, 2)} ms")

        # depth_maps[0] has shape (1, H, W) with values in [0, 1]
        # squeeze to (H, W) and convert to uint8 for display
        depth_map = (depth_maps[0].squeeze(0) * 255).astype("uint8")
        depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
        cv2.imshow("Depth Map", depth_colored)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    set_log_level("ERROR")
    main()
