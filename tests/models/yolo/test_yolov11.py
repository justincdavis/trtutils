# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from .common import (
    DLA_ENGINES,
    GPU_ENGINES,
    yolo_results,
    yolo_run,
    yolo_run_multiple,
    yolo_swapping_preproc_results,
)


def test_yolo_11_cpu_run() -> None:
    """Test GPU engine runs with CPU preproc."""
    yolo_run(11, preprocessor="cpu", use_dla=False)


def test_yolo_11_cuda_run() -> None:
    """Test GPU engine runs with CUDA preproc."""
    yolo_run(11, preprocessor="cuda", use_dla=False)


def test_yolo_11_dla_cpu_run() -> None:
    """Test DLA engine runs with CPU preproc."""
    yolo_run(11, preprocessor="cpu", use_dla=True)


def test_yolo_11_dla_cuda_run() -> None:
    """Test DLA engine runs with CUDA preproc."""
    yolo_run(11, preprocessor="cuda", use_dla=True)


def test_yolo_11_trt_run() -> None:
    """Test GPU engine runs with TRT preproc."""
    yolo_run(11, preprocessor="trt", use_dla=False)


def test_yolo_11_dla_trt_run() -> None:
    """Test DLA engine runs with TRT preproc."""
    yolo_run(11, preprocessor="trt", use_dla=True)


def test_yolo_11_cpu_multi_run() -> None:
    """Test multiple GPU engines runs with CPU preproc."""
    yolo_run_multiple(11, preprocessor="cpu", count=GPU_ENGINES, use_dla=False)


def test_yolo_11_cuda_multi_run() -> None:
    """Test multiple GPU engines runs with CUDA preproc."""
    yolo_run_multiple(11, preprocessor="cuda", count=GPU_ENGINES, use_dla=False)


def test_yolo_11_dla_cpu_multi_run() -> None:
    """Test multiple DLA engines runs with CPU preproc."""
    yolo_run_multiple(11, preprocessor="cpu", count=DLA_ENGINES, use_dla=True)


def test_yolo_11_dla_cuda_multi_run() -> None:
    """Test multiple DLA engines runs with CUDA preproc."""
    yolo_run_multiple(11, preprocessor="cuda", count=DLA_ENGINES, use_dla=True)


def test_yolo_11_trt_multi_run() -> None:
    """Test multiple GPU engines runs with TRT preproc."""
    yolo_run_multiple(11, preprocessor="trt", count=GPU_ENGINES, use_dla=False)


def test_yolo_11_dla_trt_multi_run() -> None:
    """Test multiple DLA engines runs with TRT preproc."""
    yolo_run_multiple(11, preprocessor="trt", count=DLA_ENGINES, use_dla=True)


def test_yolo_11_preproc_cpu_results() -> None:
    """Test GPU engine with CPU preproc has valid results."""
    yolo_results(11, preprocessor="cpu", use_dla=False)


def test_yolo_11_preproc_cuda_results() -> None:
    """Test GPU engine with CUDa preproc has valid results."""
    yolo_results(11, preprocessor="cuda", use_dla=False)


def test_yolo_11_dla_preproc_cpu_results() -> None:
    """Test DLA engine with CPU preproc has valid results."""
    yolo_results(11, preprocessor="cpu", use_dla=True)


def test_yolo_11_dla_preproc_cuda_results() -> None:
    """Test DLA engine with CUDA preproc has valid results."""
    yolo_results(11, preprocessor="cuda", use_dla=True)


def test_yolo_11_preproc_trt_results() -> None:
    """Test GPU engine with TRT preproc has valid results."""
    yolo_results(11, preprocessor="trt", use_dla=False)


def test_yolo_11_dla_preproc_trt_results() -> None:
    """Test DLA engine with TRT preproc has valid results."""
    yolo_results(11, preprocessor="trt", use_dla=True)


def test_yolo_11_swapping_preproc_results() -> None:
    """Test swapping the preprocessing method at runtime and check results."""
    yolo_swapping_preproc_results(11, use_dla=False)


def test_yolo_11_swapping_preproc_results_dla() -> None:
    """Test swapping the preprocessing method at runtime and check results with DLA."""
    yolo_swapping_preproc_results(11, use_dla=True)


if __name__ == "__main__":
    test_yolo_11_preproc_cuda_results()
