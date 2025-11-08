# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from .common import download_with_args


def test_yolov7_download_no_args():
    """Test downloading YOLOv7 model with no arguments."""
    download_with_args("yolov7t")


def test_yolov7_download_image_sizes():
    """Test downloading YOLOv7 model with different image sizes."""
    download_with_args("yolov7t", imgsz=320)
    download_with_args("yolov7t", imgsz=640)
    download_with_args("yolov7t", imgsz=960)


def test_yolov7_download_opsets():
    """Test downloading YOLOv7 model with different opsets."""
    download_with_args("yolov7t", opset=13)
    download_with_args("yolov7t", opset=14)
    download_with_args("yolov7t", opset=15)
    download_with_args("yolov7t", opset=16)
    download_with_args("yolov7t", opset=17)
    download_with_args("yolov7t", opset=18)
    download_with_args("yolov7t", opset=19)
    download_with_args("yolov7t", opset=20)
