# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import time

import cv2
import numpy as np

from trtutils.impls.yolo import CPUPreprocessor, CUDAPreprocessor, TRTPreprocessor

from .paths import HORSE_IMAGE_PATH, IMAGE_PATHS

CUDA_MAG_BOUNDS = 0.01
TRT_MAG_BOUNDS = 0.01


def test_cpu_preproc_loads() -> None:
    """Test if the CPUPreprocessor loads."""
    preproc = CPUPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))
    assert preproc


def test_cuda_preproc_loads() -> None:
    """Test if the CPUPreprocessor loads."""
    preproc = CUDAPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))
    assert preproc


def test_trt_preproc_loads() -> None:
    """Test if the CPUPreprocessor loads."""
    preproc = TRTPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))
    assert preproc


def test_cpu_preproc_duplicate() -> None:
    """Checks that the same data will give same results with CPU."""
    preproc = CPUPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))
    img = cv2.imread(HORSE_IMAGE_PATH)
    result1 = preproc.preprocess(img)[0]
    result2 = preproc.preprocess(img)[0]
    assert result1.all() == result2.all()


def test_cuda_preproc_duplicate() -> None:
    """Checks that the same data will give same results with CUDA."""
    preproc = CUDAPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))
    img = cv2.imread(HORSE_IMAGE_PATH)
    result1 = preproc.preprocess(img)[0]
    result2 = preproc.preprocess(img)[0]
    assert result1.all() == result2.all()


def test_trt_preproc_duplicate() -> None:
    """Checks that the same data will give same results with TRT."""
    preproc = TRTPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))
    img = cv2.imread(HORSE_IMAGE_PATH)
    result1 = preproc.preprocess(img)[0]
    result2 = preproc.preprocess(img)[0]
    assert result1.all() == result2.all()


def _assess_parity(
    preproc1: CPUPreprocessor | CUDAPreprocessor | TRTPreprocessor,
    tag1: str,
    preproc2: CPUPreprocessor | CUDAPreprocessor | TRTPreprocessor,
    tag2: str,
    method: str,
) -> None:
    for img_path in IMAGE_PATHS:
        img = cv2.imread(img_path)
        result1, ratios1, padding1 = preproc1.preprocess(img, resize=method)
        result2, ratios2, padding2 = preproc2.preprocess(img, resize=method)
        assert ratios1 == ratios2
        assert padding1 == padding2
        assert result1.shape == result2.shape
        assert result1.dtype == result2.dtype
        cpu_mean = np.mean(result1)
        cuda_mean = np.mean(result2)
        assert cpu_mean - CUDA_MAG_BOUNDS <= cuda_mean <= cpu_mean + CUDA_MAG_BOUNDS, (
            f"{tag1}: {cpu_mean:.3f}, {tag2}: {cuda_mean:.3f}"
        )
        diff_mask = np.any(result1 != result2, axis=-1)
        avg_diff = np.mean(np.abs(result1[diff_mask] - result2[diff_mask]))
        assert avg_diff < 1.0


def test_cuda_parity_linear() -> None:
    """Test the results of the CUDA preprocessor againist the CPU preprocessor."""
    cpu = CPUPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))
    cuda = CUDAPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))

    _assess_parity(cpu, "CPU", cuda, "CUDA", "linear")


def test_cuda_parity_letterbox() -> None:
    """Test the results of the CUDA preprocessor againist the CPU preprocessor."""
    cpu = CPUPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))
    cuda = CUDAPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))

    _assess_parity(cpu, "CPU", cuda, "CUDA", "letterbox")


def test_trt_parity_linear() -> None:
    """Test the results of the TRT preprocessor againist the CPU preprocessor."""
    cpu = CPUPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))
    trt = TRTPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))

    _assess_parity(cpu, "CPU", trt, "TRT", "linear")


def test_trt_parity_letterbox() -> None:
    """Test the results of the TRT preprocessor againist the CPU preprocessor."""
    cpu = CPUPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))
    trt = TRTPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))

    _assess_parity(cpu, "CPU", trt, "TRT", "letterbox")


def test_cuda_perf() -> None:
    """Test that the CUDA preprocessor is faster than the CPU preprocessor."""
    cpu = CPUPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))
    cuda = CUDAPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))

    img = cv2.imread(HORSE_IMAGE_PATH)

    # warmup
    for _ in range(10):
        cpu.preprocess(img)
        cuda.preprocess(img)

    # measure CPU time
    def _measure(preproc: CPUPreprocessor | CUDAPreprocessor) -> float:
        profs = []
        for _ in range(10):
            t0 = time.perf_counter()
            preproc.preprocess(img)
            t1 = time.perf_counter()
            profs.append(t1 - t0)
        return np.mean(profs)

    cpu_time = _measure(cpu)
    cuda_time = _measure(cuda)

    assert cpu_time > cuda_time


def test_trt_perf() -> None:
    """Test that the TRT preprocessor is faster than the CPU preprocessor."""
    cpu = CPUPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))
    trt = TRTPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))

    img = cv2.imread(HORSE_IMAGE_PATH)

    # warmup
    for _ in range(10):
        cpu.preprocess(img)
        trt.preprocess(img)

    # measure CPU time
    def _measure(preproc: CPUPreprocessor | TRTPreprocessor) -> float:
        profs = []
        for _ in range(10):
            t0 = time.perf_counter()
            preproc.preprocess(img)
            t1 = time.perf_counter()
            profs.append(t1 - t0)
        return np.mean(profs)

    cpu_time = _measure(cpu)
    trt_time = _measure(trt)

    assert cpu_time > trt_time
