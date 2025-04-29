# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import cv2
import numpy as np

import trtutils

from .paths import HORSE_IMAGE_PATH, IMAGE_PATHS

CUDA_MAG_BOUNDS = 0.01
TRT_MAG_BOUNDS = 0.01


def test_cpu_preproc_loads() -> None:
    """Test if the CPUPreprocessor loads."""
    preproc = trtutils.impls.yolo.CPUPreprocessor(
        (640, 640), (0.0, 1.0), np.dtype(np.float32)
    )
    assert preproc


def test_cuda_preproc_loads() -> None:
    """Test if the CPUPreprocessor loads."""
    preproc = trtutils.impls.yolo.CUDAPreprocessor(
        (640, 640), (0.0, 1.0), np.dtype(np.float32)
    )
    assert preproc


def test_trt_preproc_loads() -> None:
    """Test if the CPUPreprocessor loads."""
    preproc = trtutils.impls.yolo.TRTPreprocessor(
        (640, 640), (0.0, 1.0), np.dtype(np.float32)
    )
    assert preproc


def test_cpu_preproc_duplicate() -> None:
    """Checks that the same data will give same results with CPU."""
    preproc = trtutils.impls.yolo.CPUPreprocessor(
        (640, 640), (0.0, 1.0), np.dtype(np.float32)
    )
    img = cv2.imread(HORSE_IMAGE_PATH)
    result1 = preproc.preprocess(img)[0]
    result2 = preproc.preprocess(img)[0]
    assert result1.all() == result2.all()


def test_cuda_preproc_duplicate() -> None:
    """Checks that the same data will give same results with CUDA."""
    preproc = trtutils.impls.yolo.CUDAPreprocessor(
        (640, 640), (0.0, 1.0), np.dtype(np.float32)
    )
    img = cv2.imread(HORSE_IMAGE_PATH)
    result1 = preproc.preprocess(img)[0]
    result2 = preproc.preprocess(img)[0]
    assert result1.all() == result2.all()


def test_trt_preproc_duplicate() -> None:
    """Checks that the same data will give same results with TRT."""
    preproc = trtutils.impls.yolo.TRTPreprocessor(
        (640, 640), (0.0, 1.0), np.dtype(np.float32)
    )
    img = cv2.imread(HORSE_IMAGE_PATH)
    result1 = preproc.preprocess(img)[0]
    result2 = preproc.preprocess(img)[0]
    assert result1.all() == result2.all()


def test_cuda_parity() -> None:
    """Test the results of the CUDA preprocessor againist the CPU preprocessor."""
    cpu = trtutils.impls.yolo.CPUPreprocessor(
        (640, 640), (0.0, 1.0), np.dtype(np.float32)
    )
    cuda = trtutils.impls.yolo.CUDAPreprocessor(
        (640, 640), (0.0, 1.0), np.dtype(np.float32)
    )

    for img_path in IMAGE_PATHS:
        img = cv2.imread(img_path)
        cpu_result, cpu_ratios, cpu_padding = cpu.preprocess(img, resize="linear")
        cuda_result, cuda_ratios, cuda_padding = cuda.preprocess(img, resize="linear")
        assert cpu_ratios == cuda_ratios
        assert cpu_padding == cuda_padding
        assert cpu_result.shape == cuda_result.shape
        assert cpu_result.dtype == cuda_result.dtype
        cpu_mean = np.mean(cpu_result)
        cuda_mean = np.mean(cuda_result)
        assert cpu_mean - CUDA_MAG_BOUNDS <= cuda_mean <= cpu_mean + CUDA_MAG_BOUNDS, (
            f"CPU: {cpu_mean:.3f}, CUDA: {cuda_mean:.3f}"
        )
        diff_mask = np.any(cpu_result != cuda_result, axis=-1)
        avg_diff = np.mean(np.abs(cpu_result[diff_mask] - cuda_result[diff_mask]))
        assert avg_diff < 1.0


def test_trt_parity() -> None:
    """Test the results of the TRT preprocessor againist the CPU preprocessor."""
    cpu = trtutils.impls.yolo.CPUPreprocessor(
        (640, 640), (0.0, 1.0), np.dtype(np.float32)
    )
    trt = trtutils.impls.yolo.TRTPreprocessor(
        (640, 640), (0.0, 1.0), np.dtype(np.float32)
    )

    for img_path in IMAGE_PATHS:
        img = cv2.imread(img_path)
        cpu_result, cpu_ratios, cpu_padding = cpu.preprocess(img, resize="linear")
        trt_result, trt_ratios, trt_padding = trt.preprocess(img, resize="linear")
        assert cpu_ratios == trt_ratios
        assert cpu_padding == trt_padding
        assert cpu_result.shape == trt_result.shape
        assert cpu_result.dtype == trt_result.dtype
        cpu_mean = np.mean(cpu_result)
        trt_mean = np.mean(trt_result)
        assert cpu_mean - TRT_MAG_BOUNDS <= trt_mean <= cpu_mean + TRT_MAG_BOUNDS, (
            f"CPU: {cpu_mean:.3f}, TRT: {trt_mean:.3f}"
        )
        diff_mask = np.any(cpu_result != trt_result, axis=-1)
        avg_diff = np.mean(np.abs(cpu_result[diff_mask] - trt_result[diff_mask]))
        assert avg_diff < 1.0
