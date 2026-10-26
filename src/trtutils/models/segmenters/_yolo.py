# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import ClassVar

from trtutils.image._segmenter import Segmenter
from trtutils.models._model import Model


class YOLOv8Seg(Segmenter, Model):
    """Alias of Segmenter with default args for YOLOv8-Seg."""

    _model_type = "yolov8_seg"
    _friendly_name = "YOLOv8-Seg"
    _default_imgsz = 640
    _input_tensors: ClassVar[list[tuple[str, str]]] = [("images", "image")]


class YOLOv11Seg(Segmenter, Model):
    """Alias of Segmenter with default args for YOLOv11-Seg."""

    _model_type = "yolov11_seg"
    _friendly_name = "YOLOv11-Seg"
    _default_imgsz = 640
    _input_tensors: ClassVar[list[tuple[str, str]]] = [("images", "image")]


class YOLOv26Seg(Segmenter, Model):
    """
    Alias of Segmenter with default args for YOLOv26-Seg.

    YOLOv26 task heads export raw (``end2end=False``) and are decoded with NMS, so
    results can differ from ultralytics' default NMS-free ``.pt`` predict.
    """

    _model_type = "yolov26_seg"
    _friendly_name = "YOLOv26-Seg"
    _default_imgsz = 640
    _input_tensors: ClassVar[list[tuple[str, str]]] = [("images", "image")]
