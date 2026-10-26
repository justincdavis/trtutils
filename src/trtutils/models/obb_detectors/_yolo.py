# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import ClassVar

from trtutils.image._obb_detector import OBBDetector
from trtutils.models._model import Model


class YOLOv8OBB(OBBDetector, Model):
    """
    Alias of OBBDetector with default args for YOLOv8-OBB.

    These OBB weights are DOTA-trained (aerial imagery), not COCO.
    """

    _model_type = "yolov8_obb"
    _friendly_name = "YOLOv8-OBB"
    _default_imgsz = 1024
    _imgsz_divisor = 32
    _input_tensors: ClassVar[list[tuple[str, str]]] = [("images", "image")]


class YOLOv11OBB(OBBDetector, Model):
    """
    Alias of OBBDetector with default args for YOLOv11-OBB.

    These OBB weights are DOTA-trained (aerial imagery), not COCO.
    """

    _model_type = "yolov11_obb"
    _friendly_name = "YOLOv11-OBB"
    _default_imgsz = 1024
    _imgsz_divisor = 32
    _input_tensors: ClassVar[list[tuple[str, str]]] = [("images", "image")]


class YOLOv26OBB(OBBDetector, Model):
    """
    Alias of OBBDetector with default args for YOLOv26-OBB.

    These OBB weights are DOTA-trained (aerial imagery), not COCO.

    YOLOv26 task heads export raw (``end2end=False``) and are decoded with NMS, so
    results can differ from ultralytics' default NMS-free ``.pt`` predict.
    """

    _model_type = "yolov26_obb"
    _friendly_name = "YOLOv26-OBB"
    _default_imgsz = 1024
    _imgsz_divisor = 32
    _input_tensors: ClassVar[list[tuple[str, str]]] = [("images", "image")]
