# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Public type aliases for image model results.

Aliases
-------
:data:`Bbox`
    A bounding box as (x1, y1, x2, y2).
:data:`Detection`
    One detection as (bbox, score, class_id).
:data:`Classification`
    One classification as (class_id, confidence).
:data:`HandInteraction`
    One hand-object interaction as (hand, first, second, side, contact).
:data:`TextBox`
    One detected text region as (box, score).
:data:`Text`
    One recognized text line as (text, confidence).
:data:`TextLine`
    One detected and recognized text line as (box, text, score).

Examples
--------
>>> from trtutils.image.types import Detection
>>> def largest(dets: list[Detection]) -> Detection: ...

"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from typing_extensions import TypeAlias

#: bounding box as (x1, y1, x2, y2) in original image pixels
Bbox: TypeAlias = Tuple[int, int, int, int]

#: detection as (bbox, score, class_id)
Detection: TypeAlias = Tuple[Bbox, float, int]

#: classification as (class_id, confidence)
Classification: TypeAlias = Tuple[int, float]

#: hand interaction as ((hand bbox, score), (first object bbox, score) | None,
#: (second object bbox, score) | None, side | None, contact | None)
HandInteraction: TypeAlias = Tuple[
    Tuple[Bbox, float],
    Optional[Tuple[Bbox, float]],
    Optional[Tuple[Bbox, float]],
    Optional[int],
    Optional[int],
]

#: text region as (box, score), box is the (4, 2) float32 corner points (TL, TR, BR, BL)
TextBox: TypeAlias = Tuple[np.ndarray, float]

#: recognized text as (text, confidence)
Text: TypeAlias = Tuple[str, float]

#: text line as (box, text, score), box as in :data:`TextBox`
TextLine: TypeAlias = Tuple[np.ndarray, str, float]

__all__ = [
    "Bbox",
    "Classification",
    "Detection",
    "HandInteraction",
    "Text",
    "TextBox",
    "TextLine",
]
