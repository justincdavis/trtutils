# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Webcam depth estimator demo: pick any supported depth model by name."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, cast

import cv2
import cv2ext
import numpy as np
from _common import DEPTH_LOOKUP, prepare_engine, webcam_loop

if TYPE_CHECKING:
    from trtutils.image import DepthEstimator

_DEPTH_NDIM_3D = 3
_DEPTH_EPS = 1e-6


def _depth_to_color(depth: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Normalize, colorize, and resize a depth map to match the frame."""
    d = depth.astype(np.float32)
    if d.ndim == _DEPTH_NDIM_3D and d.shape[0] == 1:
        d = d[0]
    elif d.ndim == _DEPTH_NDIM_3D and d.shape[-1] == 1:
        d = d[..., 0]
    d_min = float(d.min())
    d_max = float(d.max())
    if d_max - d_min < _DEPTH_EPS:
        normed = np.zeros_like(d, dtype=np.uint8)
    else:
        normed = ((d - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(normed, cv2.COLORMAP_INFERNO)
    h, w = out_hw
    if color.shape[:2] != (h, w):
        color = cv2.resize(color, (w, h), interpolation=cv2.INTER_LINEAR)
    return color


def _main(source: int, model_name: str, *, verbose: bool) -> None:
    cls_raw, engine_path = prepare_engine(
        model_name,
        DEPTH_LOOKUP,
        task="depth estimator",
        verbose=verbose,
    )
    cls = cast("type[DepthEstimator]", cls_raw)
    estimator = cls(
        engine_path,
        warmup=True,
        warmup_iterations=10,
        preprocessor="trt",
        resize_method="linear",
    )

    def infer(frame: np.ndarray) -> np.ndarray:
        return estimator.end2end([frame])[0]

    def render(frame: np.ndarray, depth: np.ndarray, fps: float) -> np.ndarray:
        depth_color = _depth_to_color(depth, frame.shape[:2])
        canvas = np.hstack([frame, depth_color])
        cv2ext.image.draw.text(canvas, f"{fps:.2f} FPS", (10, 30))
        cv2ext.image.draw.text(canvas, f"model: {model_name}", (10, 60))
        return canvas

    webcam_loop(source, f"Depth — {model_name}", infer, render)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Webcam Depth Estimator Demo")
    parser.add_argument(
        "--model",
        required=True,
        help="Depth model name (e.g. depth_anything_v2_small).",
    )
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _main(args.source, args.model, verbose=args.verbose)
