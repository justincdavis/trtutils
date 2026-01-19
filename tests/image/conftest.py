# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import numpy as np
import pytest

from trtutils.image.preprocessors import (
    CPUPreprocessor,
    CUDAPreprocessor,
    TRTPreprocessor,
)

PREPROC_SIZE = (640, 640)
PREPROC_RANGE = (0.0, 1.0)
PREPROC_DTYPE = np.dtype(np.float32)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@pytest.fixture(params=["cpu", "cuda", "trt"])
def preprocessor_type(request) -> str:
    return request.param


@pytest.fixture
def make_preprocessor():
    def _make(
        ptype: str,
        *,
        mean: tuple[float, ...] | None = None,
        std: tuple[float, ...] | None = None,
        batch_size: int = 4,
    ):
        if ptype == "cpu":
            return CPUPreprocessor(PREPROC_SIZE, PREPROC_RANGE, PREPROC_DTYPE, mean=mean, std=std)
        if ptype == "cuda":
            return CUDAPreprocessor(PREPROC_SIZE, PREPROC_RANGE, PREPROC_DTYPE, mean=mean, std=std)
        if ptype == "trt":
            return TRTPreprocessor(
                PREPROC_SIZE, PREPROC_RANGE, PREPROC_DTYPE, mean=mean, std=std, batch_size=batch_size
            )
        raise ValueError(f"Unknown preprocessor type: {ptype}")

    return _make


@pytest.fixture(params=["linear", "letterbox"])
def resize_method(request) -> str:
    return request.param


@pytest.fixture
def make_ratios_padding():
    def _make(batch_size: int) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        ratios = [(1.0, 1.0) for _ in range(batch_size)]
        padding = [(0.0, 0.0) for _ in range(batch_size)]
        return ratios, padding

    return _make


@pytest.fixture
def make_yolov10_output():
    def _make(batch_size: int, num_dets: int = 10) -> list[np.ndarray]:
        output = np.zeros((batch_size, 300, 6), dtype=np.float32)
        for b in range(batch_size):
            for i in range(num_dets):
                offset = b * 50
                output[b, i] = [
                    100 + i * 10 + offset,
                    100 + i * 10 + offset,
                    200 + i * 10 + offset,
                    200 + i * 10 + offset,
                    0.9 - i * 0.05,
                    i % 10,
                ]
        return [output]

    return _make


@pytest.fixture
def make_efficient_nms_output():
    def _make(batch_size: int, num_dets: int = 10) -> list[np.ndarray]:
        max_dets = 100
        num_dets_arr = np.full((batch_size,), num_dets, dtype=np.int32)
        bboxes = np.zeros((batch_size, max_dets, 4), dtype=np.float32)
        scores = np.zeros((batch_size, max_dets), dtype=np.float32)
        class_ids = np.zeros((batch_size, max_dets), dtype=np.float32)
        for b in range(batch_size):
            offset = b * 50
            for i in range(num_dets):
                bboxes[b, i] = [
                    100 + i * 10 + offset,
                    100 + i * 10 + offset,
                    200 + i * 10 + offset,
                    200 + i * 10 + offset,
                ]
                scores[b, i] = 0.9 - i * 0.05
                class_ids[b, i] = i % 10
        return [num_dets_arr, bboxes, scores, class_ids]

    return _make


@pytest.fixture
def make_rfdetr_output():
    def _make(
        batch_size: int, num_queries: int = 300, num_classes: int = 80, num_dets: int = 10
    ) -> list[np.ndarray]:
        dets = np.zeros((batch_size, num_queries, 4), dtype=np.float32)
        labels = np.full((batch_size, num_queries, num_classes), -10.0, dtype=np.float32)
        for b in range(batch_size):
            for i in range(num_dets):
                cx = (150 + i * 10 + b * 30) / 640.0
                cy = (150 + i * 10 + b * 30) / 640.0
                w = 100 / 640.0
                h = 100 / 640.0
                dets[b, i] = [cx, cy, w, h]
                class_idx = i % num_classes
                labels[b, i, class_idx] = 5.0 - i * 0.3
        return [dets, labels]

    return _make


@pytest.fixture
def make_detr_output():
    def _make(batch_size: int, num_queries: int = 300, num_dets: int = 10) -> list[np.ndarray]:
        scores = np.zeros((batch_size, num_queries), dtype=np.float32)
        labels = np.zeros((batch_size, num_queries), dtype=np.float32)
        boxes = np.zeros((batch_size, num_queries, 4), dtype=np.float32)
        for b in range(batch_size):
            offset = b * 50
            for i in range(num_dets):
                scores[b, i] = 0.9 - i * 0.05
                labels[b, i] = i % 10
                boxes[b, i] = [
                    100 + i * 10 + offset,
                    100 + i * 10 + offset,
                    200 + i * 10 + offset,
                    200 + i * 10 + offset,
                ]
        return [scores, labels, boxes]

    return _make


@pytest.fixture
def make_classification_output():
    def _make(batch_size: int, num_classes: int = 1000) -> list[np.ndarray]:
        output = np.random.randn(batch_size, num_classes).astype(np.float32)
        for b in range(batch_size):
            output[b, b % num_classes] = 10.0
            output[b, (b + 1) % num_classes] = 8.0
        return [output]

    return _make
