# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from .common import yolo_pagelocked_perf


def test_yolo_7_pagelocked_perf() -> None:
    """Test the performance of the YOLOv7 model."""
    yolo_pagelocked_perf(7, use_dla=False)


def test_yolo_7_pagelocked_perf_dla() -> None:
    """Test the performance of the YOLOv7 model with DLA."""
    yolo_pagelocked_perf(7, use_dla=True)


def test_yolo_8_pagelocked_perf() -> None:
    """Test the performance of the YOLOv8 model."""
    yolo_pagelocked_perf(8, use_dla=False)


def test_yolo_8_pagelocked_perf_dla() -> None:
    """Test the performance of the YOLOv8 model with DLA."""
    yolo_pagelocked_perf(8, use_dla=True)


def test_yolo_9_pagelocked_perf() -> None:
    """Test the performance of the YOLOv9 model."""
    yolo_pagelocked_perf(9, use_dla=False)


def test_yolo_9_pagelocked_perf_dla() -> None:
    """Test the performance of the YOLOv9 model with DLA."""
    yolo_pagelocked_perf(9, use_dla=True)


def test_yolo_10_pagelocked_perf() -> None:
    """Test the performance of the YOLOv10 model."""
    yolo_pagelocked_perf(10, use_dla=False)


def test_yolo_10_pagelocked_perf_dla() -> None:
    """Test the performance of the YOLOv10 model with DLA."""
    yolo_pagelocked_perf(10, use_dla=True)


def test_yolo_11_pagelocked_perf() -> None:
    """Test the performance of the YOLOv11 model."""
    yolo_pagelocked_perf(11, use_dla=False)


def test_yolo_11_pagelocked_perf_dla() -> None:
    """Test the performance of the YOLOv11 model with DLA."""
    yolo_pagelocked_perf(11, use_dla=True)


def test_yolo_12_pagelocked_perf() -> None:
    """Test the performance of the YOLOv12 model."""
    yolo_pagelocked_perf(12, use_dla=False)


def test_yolo_12_pagelocked_perf_dla() -> None:
    """Test the performance of the YOLOv12 model with DLA."""
    yolo_pagelocked_perf(12, use_dla=True)


def test_yolo_13_pagelocked_perf() -> None:
    """Test the performance of the YOLOv13 model."""
    yolo_pagelocked_perf(13, use_dla=False)


def test_yolo_13_pagelocked_perf_dla() -> None:
    """Test the performance of the YOLOv13 model with DLA."""
    yolo_pagelocked_perf(13, use_dla=True)


def test_yolox_pagelocked_perf() -> None:
    """Test the performance of the YOLOX model."""
    yolo_pagelocked_perf(0, use_dla=False)


def test_yolox_pagelocked_perf_dla() -> None:
    """Test the performance of the YOLOX model with DLA."""
    yolo_pagelocked_perf(0, use_dla=True)
