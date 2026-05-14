# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Webcam detector demo: pick any supported detector by name."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, cast

import cv2ext
from _common import DETECTOR_LOOKUP, prepare_engine, webcam_loop

if TYPE_CHECKING:
    import numpy as np

    from trtutils.image import Detector


def _main(source: int, model_name: str, *, verbose: bool) -> None:
    cls_raw, engine_path = prepare_engine(
        model_name,
        DETECTOR_LOOKUP,
        task="detector",
        verbose=verbose,
    )
    cls = cast("type[Detector]", cls_raw)
    detector = cls(
        engine_path,
        warmup=True,
        warmup_iterations=10,
        preprocessor="trt",
        resize_method="letterbox",
    )

    def infer(frame: np.ndarray) -> list[tuple]:
        return detector.end2end([frame])[0]

    def render(frame: np.ndarray, dets: list[tuple], fps: float) -> np.ndarray:
        canvas = cv2ext.bboxes.draw_bboxes(frame, [bbox for bbox, _, _ in dets])
        cv2ext.image.draw.text(canvas, f"{fps:.2f} FPS", (10, 30))
        cv2ext.image.draw.text(canvas, f"model: {model_name}", (10, 60))
        return canvas

    webcam_loop(source, f"Detector — {model_name}", infer, render)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Webcam Detector Demo")
    parser.add_argument(
        "--model",
        required=True,
        help="Detector model name (e.g. yolov10n, rtdetrv2_r18, dfine_n, rfdetr_n).",
    )
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _main(args.source, args.model, verbose=args.verbose)
