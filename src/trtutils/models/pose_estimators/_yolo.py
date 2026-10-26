# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import ClassVar

from trtutils.image._pose_estimator import PoseEstimator
from trtutils.models._model import Model


class YOLOv8Pose(PoseEstimator, Model):
    """Alias of PoseEstimator with default args for YOLOv8-Pose."""

    _model_type = "yolov8_pose"
    _friendly_name = "YOLOv8-Pose"
    _default_imgsz = 640
    _input_tensors: ClassVar[list[tuple[str, str]]] = [("images", "image")]


class YOLOv11Pose(PoseEstimator, Model):
    """Alias of PoseEstimator with default args for YOLOv11-Pose."""

    _model_type = "yolov11_pose"
    _friendly_name = "YOLOv11-Pose"
    _default_imgsz = 640
    _input_tensors: ClassVar[list[tuple[str, str]]] = [("images", "image")]


class YOLOv26Pose(PoseEstimator, Model):
    """
    Alias of PoseEstimator with default args for YOLOv26-Pose.

    YOLOv26 task heads export raw (``end2end=False``) and are decoded with NMS, so
    results can differ from ultralytics' default NMS-free ``.pt`` predict.
    """

    _model_type = "yolov26_pose"
    _friendly_name = "YOLOv26-Pose"
    _default_imgsz = 640
    _input_tensors: ClassVar[list[tuple[str, str]]] = [("images", "image")]
