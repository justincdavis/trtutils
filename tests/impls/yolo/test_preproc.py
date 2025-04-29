# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import cv2
import trtutils
import numpy as np

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
        cpu_result = cpu.preprocess(img, resize="linear")
        cuda_result = cuda.preprocess(img, resize="linear")
        assert cpu_result.shape == cuda_result.shape
        assert np.allclose(cpu_result, cuda_result)
        assert cpu_result == cuda_result


def test_trt_parity() -> None:
    """Test the results of the TRT preprocessor againist the CPU preprocessor."""
    cpu = trtutils.impls.yolo.CPUPreprocessor((640, 640), (0.0, 1.0), np.float32)
    trt = trtutils.impls.yolo.TRTPreprocessor((640, 640), (0.0, 1.0), np.float32)

    for ipath in IMAGE_PATHS:
        img = cv2.imread(ipath)
        cpu_result = cpu.preprocess(img, resize="linear")
        trt_result = trt.preprocess(img, resize="linear")
        assert cpu_result.shape == trt_result.shape
        assert np.allclose(cpu_result, trt_result)
        assert cpu_result == trt_result
