# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from .common import (
    DLA_ENGINES,
    GPU_ENGINES,
    rtdetr_results,
    rtdetr_run,
    rtdetr_run_multiple,
    rtdetr_swapping_preproc_results,
)


def test_dfine_cpu_run() -> None:
    """Test GPU engine runs with CPU preproc."""
    rtdetr_run("dfine", preprocessor="cpu", use_dla=False)


def test_dfine_cuda_run() -> None:
    """Test GPU engine runs with CUDA preproc."""
    rtdetr_run("dfine", preprocessor="cuda", use_dla=False)


def test_dfine_dla_cpu_run() -> None:
    """Test DLA engine runs with CPU preproc."""
    rtdetr_run("dfine", preprocessor="cpu", use_dla=True)


def test_dfine_dla_cuda_run() -> None:
    """Test DLA engine runs with CUDA preproc."""
    rtdetr_run("dfine", preprocessor="cuda", use_dla=True)


def test_dfine_trt_run() -> None:
    """Test GPU engine runs with TRT preproc."""
    rtdetr_run("dfine", preprocessor="trt", use_dla=False)


def test_dfine_dla_trt_run() -> None:
    """Test DLA engine runs with TRT preproc."""
    rtdetr_run("dfine", preprocessor="trt", use_dla=True)


def test_dfine_cpu_multi_run() -> None:
    """Test multiple GPU engines runs with CPU preproc."""
    rtdetr_run_multiple("dfine", preprocessor="cpu", count=GPU_ENGINES, use_dla=False)


def test_dfine_cuda_multi_run() -> None:
    """Test multiple GPU engines runs with CUDA preproc."""
    rtdetr_run_multiple("dfine", preprocessor="cuda", count=GPU_ENGINES, use_dla=False)


def test_dfine_dla_cpu_multi_run() -> None:
    """Test multiple DLA engines runs with CPU preproc."""
    rtdetr_run_multiple("dfine", preprocessor="cpu", count=DLA_ENGINES, use_dla=True)


def test_dfine_dla_cuda_multi_run() -> None:
    """Test multiple DLA engines runs with CUDA preproc."""
    rtdetr_run_multiple("dfine", preprocessor="cuda", count=DLA_ENGINES, use_dla=True)


def test_dfine_trt_multi_run() -> None:
    """Test multiple GPU engines runs with TRT preproc."""
    rtdetr_run_multiple("dfine", preprocessor="trt", count=GPU_ENGINES, use_dla=False)


def test_dfine_dla_trt_multi_run() -> None:
    """Test multiple DLA engines runs with TRT preproc."""
    rtdetr_run_multiple("dfine", preprocessor="trt", count=DLA_ENGINES, use_dla=True)


def test_dfine_preproc_cpu_results() -> None:
    """Test GPU engine with CPU preproc has valid results."""
    rtdetr_results("dfine", preprocessor="cpu", use_dla=False)


def test_dfine_preproc_cuda_results() -> None:
    """Test GPU engine with CUDA preproc has valid results."""
    rtdetr_results("dfine", preprocessor="cuda", use_dla=False)


def test_dfine_dla_preproc_cpu_results() -> None:
    """Test DLA engine with CPU preproc has valid results."""
    rtdetr_results("dfine", preprocessor="cpu", use_dla=True)


def test_dfine_dla_preproc_cuda_results() -> None:
    """Test DLA engine with CUDA preproc has valid results."""
    rtdetr_results("dfine", preprocessor="cuda", use_dla=True)


def test_dfine_preproc_trt_results() -> None:
    """Test GPU engine with TRT preproc has valid results."""
    rtdetr_results("dfine", preprocessor="trt", use_dla=False)


def test_dfine_dla_preproc_trt_results() -> None:
    """Test DLA engine with TRT preproc has valid results."""
    rtdetr_results("dfine", preprocessor="trt", use_dla=True)


def test_dfine_swapping_preproc_results() -> None:
    """Test swapping the preprocessing method at runtime and check results."""
    rtdetr_swapping_preproc_results("dfine", use_dla=False)


def test_dfine_swapping_preproc_results_dla() -> None:
    """Test swapping the preprocessing method at runtime and check results with DLA."""
    rtdetr_swapping_preproc_results("dfine", use_dla=True)


if __name__ == "__main__":
    test_dfine_preproc_cuda_results()

