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
:class:`YOLO11`
    Alias for the YOLO class with args preset for YOLO11.
:class:`YOLO12`
    Alias for the YOLO class with args preset for YOLO12.
:class:`YOLO13`
    Alias for the YOLO class with args preset for YOLO13.
:class:`YOLOX`
    Alias for the YOLO class with args preset for YOLOX.
:class:`DETR`
    Alias for the Detector class with args preset for DETR.
:class:`RTDETRv1`
    Alias for the DETR class with args preset for RT-DETRv1.
:class:`RTDETRv2`
    Alias for the DETR class with args preset for RT-DETRv2.
:class:`RTDETRv3`
    Alias for the DETR class with args preset for RT-DETRv3.
:class:`DFINE`
    Alias for the DETR class with args preset for D-FINE.
:class:`DEIM`
    Alias for the DETR class with args preset for DEIM.
:class:`DEIMv2`
    Alias for the DETR class with args preset for DEIMv2.
:class:`RFDETR`
    Alias for the DETR class with args preset for RF-DETR.

"""

from __future__ import annotations

from ._detr import DEIM, DETR, DFINE, RFDETR, DEIMv2, RTDETRv1, RTDETRv2, RTDETRv3
from ._yolo import YOLO, YOLO7, YOLO8, YOLO9, YOLO10, YOLO11, YOLO12, YOLO13, YOLOX

__all__ = [
    "DEIM",
    "DETR",
    "DFINE",
    "RFDETR",
    "YOLO",
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
