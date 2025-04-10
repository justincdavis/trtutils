# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from common import (
    DLA_ENGINES,
    GPU_ENGINES,
    yolo_results,
    yolo_run,
    yolo_run_in_thread,
    yolo_run_multiple,
    yolo_run_multiple_threads,
)

# try:
#     from common import (
#         DLA_ENGINES,
#         GPU_ENGINES,
#         yolo_results,
#         yolo_run,
#         yolo_run_in_thread,
#         yolo_run_multiple,
#         yolo_run_multiple_threads,
#     )
# except ModuleNotFoundError:
#     from .common import (
#         DLA_ENGINES,
#         GPU_ENGINES,
#         yolo_results,
#         yolo_run,
#         yolo_run_in_thread,
#         yolo_run_multiple,
#         yolo_run_multiple_threads,
#     )


def test_yolo_7_cpu_run() -> None:
    """Test GPU engine runs with CPU preproc."""
    yolo_run(7, preprocessor="cpu", use_dla=False)


def test_yolo_7_cuda_run() -> None:
    """Test GPU engine runs with CUDA preproc."""
    yolo_run(7, preprocessor="cuda", use_dla=False)


def test_yolo_7_dla_cpu_run() -> None:
    """Test DLA engine runs with CPU preproc."""
    yolo_run(7, preprocessor="cpu", use_dla=True)


def test_yolo_7_dla_cuda_run() -> None:
    """Test DLA engine runs with CUDA preproc."""
    yolo_run(7, preprocessor="cuda", use_dla=True)


def test_yolo_7_cpu_thread_run() -> None:
    """Test GPU engine runs with CPU preproc in a thread."""
    yolo_run_in_thread(7, preprocessor="cpu", use_dla=False)


def test_yolo_7_cuda_thread_run() -> None:
    """Test GPU engine runs with CUDA preproc in a thread."""
    yolo_run_in_thread(7, preprocessor="cuda", use_dla=False)


def test_yolo_7_dla_cpu_thread_run() -> None:
    """Test DLA engine runs with CPU preproc in a thread."""
    yolo_run_in_thread(7, preprocessor="cpu", use_dla=True)


def test_yolo_7_dla_cuda_thread_run() -> None:
    """Test DLA engine runs with CUDA preproc in a thread."""
    yolo_run_in_thread(7, preprocessor="cuda", use_dla=True)


def test_yolo_7_cpu_multi_run() -> None:
    """Test multiple GPU engines runs with CPU preproc."""
    yolo_run_multiple(7, preprocessor="cpu", count=GPU_ENGINES, use_dla=False)


def test_yolo_7_cuda_multi_run() -> None:
    """Test multiple GPU engines runs with CUDA preproc."""
    yolo_run_multiple(7, preprocessor="cuda", count=GPU_ENGINES, use_dla=False)


def test_yolo_7_dla_cpu_multi_run() -> None:
    """Test multiple DLA engines runs with CPU preproc."""
    yolo_run_multiple(7, preprocessor="cpu", count=DLA_ENGINES, use_dla=True)


def test_yolo_7_dla_cuda_multi_run() -> None:
    """Test multiple DLA engines runs with CUDA preproc."""
    yolo_run_multiple(7, preprocessor="cuda", count=DLA_ENGINES, use_dla=True)


def test_yolo_7_cpu_multi_thread_run() -> None:
    """Test multiple GPU engines runs with CPU preproc in multiple threads."""
    yolo_run_multiple_threads(7, preprocessor="cpu", count=GPU_ENGINES, use_dla=False)


def test_yolo_7_cuda_multi_thread_run() -> None:
    """Test multiple GPU engines runs with CUDA preproc in multiple threads."""
    yolo_run_multiple_threads(7, preprocessor="cuda", count=GPU_ENGINES, use_dla=False)


def test_yolo_7_dla_cpu_multi_thread_run() -> None:
    """Test multiple DLA engines runs with CPU preproc in multiple threads."""
    yolo_run_multiple_threads(7, preprocessor="cpu", count=DLA_ENGINES, use_dla=True)


def test_yolo_7_dla_cuda_multi_thread_run() -> None:
    """Test multiple DLA engines runs with CUDA preproc in multiple threads."""
    yolo_run_multiple_threads(7, preprocessor="cuda", count=DLA_ENGINES, use_dla=True)


def test_yolo_7_preproc_cpu_results() -> None:
    """Test GPU engine with CPU preproc has valid results."""
    yolo_results(7, preprocessor="cpu", use_dla=False)


def test_yolo_7_preproc_cuda_results() -> None:
    """Test GPU engine with CUDa preproc has valid results."""
    yolo_results(7, preprocessor="cuda", use_dla=False)


def test_yolo_7_dla_preproc_cpu_results() -> None:
    """Test DLA engine with CPU preproc has valid results."""
    yolo_results(7, preprocessor="cpu", use_dla=True)


def test_yolo_7_dla_preproc_cuda_results() -> None:
    """Test DLA engine with CUDA preproc has valid results."""
    yolo_results(7, preprocessor="cuda", use_dla=True)


if __name__ == "__main__":
    test_yolo_7_preproc_cuda_results()
