# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Implementations of various deep learning models.

Classes
-------
:class:`YOLO`
    Alias for the Detector class with args preset for YOLO.
:class:`YOLO7`
    Alias for the YOLO class with args preset for YOLO7.
:class:`YOLO8`
    Alias for the YOLO class with args preset for YOLO8.
:class:`YOLO9`
    Alias for the YOLO class with args preset for YOLO9.
:class:`YOLO10`
    Alias for the YOLO class with args preset for YOLO10.
:class:`YOLOX`
    Alias for the YOLO class with args preset for YOLOX.

"""

from __future__ import annotations

from ._yolo import YOLO, YOLO7, YOLO8, YOLO9, YOLO10, YOLOX

__all__ = [
    "YOLO",
    "YOLO7",
    "YOLO8",
    "YOLO9",
    "YOLO10",
    "YOLOX",
]
