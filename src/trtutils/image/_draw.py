# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Helpers for drawing detections on images."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

_RED = (0, 0, 255)


def draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = _RED,
    thickness: int = 2,
) -> np.ndarray:
    """Draw text with its bottom-left corner at origin, in place, and return the image."""
    return cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_bboxes(
    image: np.ndarray,
    bboxes: Sequence[tuple[int, int, int, int]],
    scores: Sequence[float] | None = None,
    classes: Sequence[int] | None = None,
    color: tuple[int, int, int] = _RED,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw (x1, y1, x2, y2) boxes on an image, in place, and return the image.

    When scores and/or classes are given, each box is tagged "class: score" above its top-left.
    """
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        tag = "" if scores is None else f"{scores[i]:.2f}"
        if classes is not None:
            tag = f"{classes[i]}: {tag}"
        if tag:
            draw_text(image, tag, (x1, y1), color, thickness=1)
    return image
