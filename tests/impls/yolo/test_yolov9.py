# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from .common import (
    DLA_ENGINES,
    GPU_ENGINES,
    yolo_results,
    yolo_run,
    yolo_run_in_thread,
    yolo_run_multiple,
    yolo_run_multiple_threads,
    yolo_swapping_preproc_results,
)


def test_yolo_9_cpu_run() -> None:
    """Test GPU engine runs with CPU preproc."""
    yolo_run(9, preprocessor="cpu", use_dla=False)


def test_yolo_9_cuda_run() -> None:
    """Test GPU engine runs with CUDA preproc."""
    yolo_run(9, preprocessor="cuda", use_dla=False)


def test_yolo_9_dla_cpu_run() -> None:
    """Test DLA engine runs with CPU preproc."""
    yolo_run(9, preprocessor="cpu", use_dla=True)


def test_yolo_9_dla_cuda_run() -> None:
    """Test DLA engine runs with CUDA preproc."""
    yolo_run(9, preprocessor="cuda", use_dla=True)


def test_yolo_9_trt_run() -> None:
    """Test GPU engine runs with TRT preproc."""
    yolo_run(9, preprocessor="trt", use_dla=False)


def test_yolo_9_dla_trt_run() -> None:
    """Test DLA engine runs with TRT preproc."""
    yolo_run(9, preprocessor="trt", use_dla=True)


def test_yolo_9_cpu_thread_run() -> None:
    """Test GPU engine runs with CPU preproc in a thread."""
    yolo_run_in_thread(9, preprocessor="cpu", use_dla=False)


def test_yolo_9_cuda_thread_run() -> None:
    """Test GPU engine runs with CUDA preproc in a thread."""
    yolo_run_in_thread(9, preprocessor="cuda", use_dla=False)


def test_yolo_9_dla_cpu_thread_run() -> None:
    """Test DLA engine runs with CPU preproc in a thread."""
    yolo_run_in_thread(9, preprocessor="cpu", use_dla=True)


def test_yolo_9_dla_cuda_thread_run() -> None:
    """Test DLA engine runs with CUDA preproc in a thread."""
    yolo_run_in_thread(9, preprocessor="cuda", use_dla=True)


def test_yolo_9_trt_thread_run() -> None:
    """Test GPU engine runs with TRT preproc in a thread."""
    yolo_run_in_thread(9, preprocessor="trt", use_dla=False)


def test_yolo_9_dla_trt_thread_run() -> None:
    """Test DLA engine runs with TRT preproc in a thread."""
    yolo_run_in_thread(9, preprocessor="trt", use_dla=True)


def test_yolo_9_cpu_multi_run() -> None:
    """Test multiple GPU engines runs with CPU preproc."""
    yolo_run_multiple(9, preprocessor="cpu", count=GPU_ENGINES, use_dla=False)


def test_yolo_9_cuda_multi_run() -> None:
    """Test multiple GPU engines runs with CUDA preproc."""
    yolo_run_multiple(9, preprocessor="cuda", count=GPU_ENGINES, use_dla=False)


def test_yolo_9_dla_cpu_multi_run() -> None:
    """Test multiple DLA engines runs with CPU preproc."""
    yolo_run_multiple(9, preprocessor="cpu", count=DLA_ENGINES, use_dla=True)


def test_yolo_9_dla_cuda_multi_run() -> None:
    """Test multiple DLA engines runs with CUDA preproc."""
    yolo_run_multiple(9, preprocessor="cuda", count=DLA_ENGINES, use_dla=True)


def test_yolo_9_trt_multi_run() -> None:
    """Test multiple GPU engines runs with TRT preproc."""
    yolo_run_multiple(9, preprocessor="trt", count=GPU_ENGINES, use_dla=False)


def test_yolo_9_dla_trt_multi_run() -> None:
    """Test multiple DLA engines runs with TRT preproc."""
    yolo_run_multiple(9, preprocessor="trt", count=DLA_ENGINES, use_dla=True)


def test_yolo_9_cpu_multi_thread_run() -> None:
    """Test multiple GPU engines runs with CPU preproc in multiple threads."""
    yolo_run_multiple_threads(9, preprocessor="cpu", count=GPU_ENGINES, use_dla=False)


def test_yolo_9_cuda_multi_thread_run() -> None:
    """Test multiple GPU engines runs with CUDA preproc in multiple threads."""
    yolo_run_multiple_threads(9, preprocessor="cuda", count=GPU_ENGINES, use_dla=False)


def test_yolo_9_dla_cpu_multi_thread_run() -> None:
    """Test multiple DLA engines runs with CPU preproc in multiple threads."""
    yolo_run_multiple_threads(9, preprocessor="cpu", count=DLA_ENGINES, use_dla=True)


def test_yolo_9_dla_cuda_multi_thread_run() -> None:
    """Test multiple DLA engines runs with CUDA preproc in multiple threads."""
    yolo_run_multiple_threads(9, preprocessor="cuda", count=DLA_ENGINES, use_dla=True)


def test_yolo_9_trt_multi_thread_run() -> None:
    """Test multiple GPU engines runs with TRT preproc in multiple threads."""
    yolo_run_multiple_threads(9, preprocessor="trt", count=GPU_ENGINES, use_dla=False)


def test_yolo_9_dla_trt_multi_thread_run() -> None:
    """Test multiple DLA engines runs with TRT preproc in multiple threads."""
    yolo_run_multiple_threads(9, preprocessor="trt", count=DLA_ENGINES, use_dla=True)


def test_yolo_9_preproc_cpu_results() -> None:
    """Test GPU engine with CPU preproc has valid results."""
    yolo_results(9, preprocessor="cpu", use_dla=False)


def test_yolo_9_preproc_cuda_results() -> None:
    """Test GPU engine with CUDa preproc has valid results."""
    yolo_results(9, preprocessor="cuda", use_dla=False)


def test_yolo_9_dla_preproc_cpu_results() -> None:
    """Test DLA engine with CPU preproc has valid results."""
    yolo_results(9, preprocessor="cpu", use_dla=True)


def test_yolo_9_dla_preproc_cuda_results() -> None:
    """Test DLA engine with CUDA preproc has valid results."""
    yolo_results(9, preprocessor="cuda", use_dla=True)


def test_yolo_9_preproc_trt_results() -> None:
    """Test GPU engine with TRT preproc has valid results."""
    yolo_results(9, preprocessor="trt", use_dla=False)


def test_yolo_9_dla_preproc_trt_results() -> None:
    """Test DLA engine with TRT preproc has valid results."""
    yolo_results(9, preprocessor="trt", use_dla=True)


def test_yolo_9_swapping_preproc_results() -> None:
    """Test swapping the preprocessing method at runtime and check results."""
    yolo_swapping_preproc_results(7, use_dla=False)


def test_yolo_9_swapping_preproc_results_dla() -> None:
    """Test swapping the preprocessing method at runtime and check results with DLA."""
    yolo_swapping_preproc_results(7, use_dla=True)


if __name__ == "__main__":
    test_yolo_9_preproc_cuda_results()
