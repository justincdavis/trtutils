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

Examples
--------
>>> from trtutils.image.types import Detection
>>> def largest(dets: list[Detection]) -> Detection: ...

"""

from __future__ import annotations

from typing import Optional, Tuple

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

__all__ = [
    "Bbox",
    "Classification",
    "Detection",
    "HandInteraction",
]
