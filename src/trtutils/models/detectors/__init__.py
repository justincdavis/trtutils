# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Detector model implementations."""

from __future__ import annotations

from ._archs import DETR, YOLO
from ._deim import DEIM, DEIMv2
from ._dfine import DFINE
from ._rfdetr import RFDETR
from ._rtdetr import RTDETRv1, RTDETRv2, RTDETRv3
from ._yolo import YOLO3, YOLO5, YOLO7, YOLO8, YOLO9, YOLO10, YOLO11, YOLO12, YOLO13, YOLOX

__all__ = [
    "DEIM",
    "DETR",
    "DFINE",
    "RFDETR",
    "YOLO",
    "YOLO3",
    "YOLO5",
    "YOLO7",
    "YOLO8",
    "YOLO9",
    "YOLO10",
    "YOLO11",
    "YOLO12",
    "YOLO13",
    "YOLOX",
    "DEIMv2",
    "RTDETRv1",
    "RTDETRv2",
    "RTDETRv3",
]
