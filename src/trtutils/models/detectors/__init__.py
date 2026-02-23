# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Detector model implementations."""

from __future__ import annotations

from ._archs import DETR, YOLO
from ._deim import DEIM, DEIMv2
from ._dfine import DFINE
from ._rfdetr import RFDETR
from ._rtdetr import RTDETRv1, RTDETRv2, RTDETRv3
from ._yolo import (
    YOLOX,
    YOLOv3,
    YOLOv5,
    YOLOv7,
    YOLOv8,
    YOLOv9,
    YOLOv10,
    YOLOv11,
    YOLOv12,
    YOLOv13,
    YOLOv26,
)

__all__ = [
    "DEIM",
    "DETR",
    "DFINE",
    "RFDETR",
    "YOLO",
    "YOLOX",
    "DEIMv2",
    "RTDETRv1",
    "RTDETRv2",
    "RTDETRv3",
    "YOLOv3",
    "YOLOv5",
    "YOLOv7",
    "YOLOv8",
    "YOLOv9",
    "YOLOv10",
    "YOLOv11",
    "YOLOv12",
    "YOLOv13",
    "YOLOv26",
]
