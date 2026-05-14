# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Webcam classifier demo: pick any supported classifier by name."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, cast

import cv2ext
from _common import CLASSIFIER_LOOKUP, prepare_engine, webcam_loop

if TYPE_CHECKING:
    import numpy as np

    from trtutils.image import Classifier

_LABELS_FILE = Path(__file__).parent / "data" / "imagenet_classes.txt"


def _load_labels() -> list[str]:
    if not _LABELS_FILE.exists():
        err_msg = f"ImageNet labels file missing: {_LABELS_FILE}"
        raise FileNotFoundError(err_msg)
    return _LABELS_FILE.read_text(encoding="utf-8").splitlines()


def _main(source: int, model_name: str, top_k: int, *, verbose: bool) -> None:
    labels = _load_labels()
    cls_raw, engine_path = prepare_engine(
        model_name,
        CLASSIFIER_LOOKUP,
        task="classifier",
        verbose=verbose,
    )
    cls = cast("type[Classifier]", cls_raw)
    classifier = cls(
        engine_path,
        warmup=True,
        warmup_iterations=10,
        preprocessor="trt",
        resize_method="linear",
    )

    def infer(frame: np.ndarray) -> list[tuple[int, float]]:
        return classifier.end2end([frame], top_k=top_k)[0]

    def render(
        frame: np.ndarray,
        preds: list[tuple[int, float]],
        fps: float,
    ) -> np.ndarray:
        canvas = frame.copy()
        cv2ext.image.draw.text(canvas, f"{fps:.2f} FPS", (10, 30))
        cv2ext.image.draw.text(canvas, f"model: {model_name}", (10, 60))
        for i, (class_id, score) in enumerate(preds):
            label = labels[class_id] if 0 <= class_id < len(labels) else f"class {class_id}"
            line = f"{label}: {score * 100:.1f}%"
            cv2ext.image.draw.text(
                canvas,
                line,
                (10, 100 + 30 * i),
                font_scale=0.7,
                thickness=2,
            )
        return canvas

    webcam_loop(source, f"Classifier — {model_name}", infer, render)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Webcam Classifier Demo")
    parser.add_argument(
        "--model",
        required=True,
        help="Classifier model name (e.g. resnet18, vit_b_16, efficientnet_b0).",
    )
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _main(args.source, args.model, args.top_k, verbose=args.verbose)
