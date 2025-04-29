# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import cv2
import numpy as np

import trtutils

from .paths import IMAGE_PATHS


def test_cpu_preproc_loads() -> None:
    """Test if the CPUPreprocessor loads."""
    preproc = trtutils.impls.yolo.CPUPreprocessor((640, 640), (0.0, 1.0), np.float32)
    assert preproc


def test_cuda_preproc_loads() -> None:
    """Test if the CPUPreprocessor loads."""
    preproc = trtutils.impls.yolo.CUDAPreprocessor((640, 640), (0.0, 1.0), np.float32)
    assert preproc


def test_trt_preproc_loads() -> None:
    """Test if the CPUPreprocessor loads."""
    preproc = trtutils.impls.yolo.TRTPreprocessor((640, 640), (0.0, 1.0), np.float32)
    assert preproc


def test_cuda_parity() -> None:
    """Test the results of the CUDA preprocessor againist the CPU preprocessor."""
    cpu = trtutils.impls.yolo.CPUPreprocessor((640, 640), (0.0, 1.0), np.float32)
    cuda = trtutils.impls.yolo.CUDAPreprocessor((640, 640), (0.0, 1.0), np.float32)

    for ipath in IMAGE_PATHS:
        img = cv2.imread(ipath)
        cpu_result, cpu_ratios, cpu_padding = cpu.preprocess(img, resize="linear")
        cuda_result, cuda_ratios, cuda_padding = cuda.preprocess(img, resize="linear")
        assert cpu_ratios == cuda_ratios
        assert cpu_padding == cuda_padding
        assert cpu_result.shape == cuda_result.shape
        assert np.allclose(cpu_result, cuda_result)
        assert cpu_result == cuda_result


def test_trt_parity() -> None:
    """Test the results of the TRT preprocessor againist the CPU preprocessor."""
    cpu = trtutils.impls.yolo.CPUPreprocessor((640, 640), (0.0, 1.0), np.float32)
    trt = trtutils.impls.yolo.TRTPreprocessor((640, 640), (0.0, 1.0), np.float32)

    for ipath in IMAGE_PATHS:
        img = cv2.imread(ipath)
        cpu_result, cpu_ratios, cpu_padding = cpu.preprocess(img, resize="linear")
        trt_result, trt_ratios, trt_padding = trt.preprocess(img, resize="linear")
        assert cpu_ratios == trt_ratios
        assert cpu_padding == trt_padding
        assert cpu_result.shape == trt_result.shape
        assert np.allclose(cpu_result, trt_result)
        assert cpu_result == trt_result


def test_trt_cuda_parity() -> None:
    """Test the results of the TRT preprocessor againist the CUDA preprocessor."""
    cuda = trtutils.impls.yolo.CUDAPreprocessor((640, 640), (0.0, 1.0), np.float32)
    trt = trtutils.impls.yolo.TRTPreprocessor((640, 640), (0.0, 1.0), np.float32)

    for ipath in IMAGE_PATHS:
        img = cv2.imread(ipath)
        cuda_result, cuda_ratios, cuda_padding = cuda.preprocess(img, resize="linear")
        trt_result, trt_ratios, trt_padding = trt.preprocess(img, resize="linear")
        assert cuda_ratios == trt_ratios
        assert cuda_padding == trt_padding
        assert cuda_result.shape == trt_result.shape
        assert np.allclose(cuda_result, trt_result)
        assert cuda_result == trt_result
