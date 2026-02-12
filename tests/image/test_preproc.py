# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="misc"
from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.conftest import HORSE_IMAGE_PATH, IMAGE_PATHS, _read_image
from trtutils.image.preprocessors import (
    CPUPreprocessor,
    CUDAPreprocessor,
    TRTPreprocessor,
)

from .conftest import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PREPROC_DTYPE,
    PREPROC_RANGE,
    PREPROC_SIZE,
)

if TYPE_CHECKING:
    from collections.abc import Callable

CUDA_MAG_BOUNDS = 0.01


class TestPreprocessorLoads:
    """Ensure preprocessors initialize correctly."""

    @pytest.mark.parametrize("ptype", ["cpu", "cuda", "trt"])
    def test_loads(
        self,
        make_preprocessor: Callable[..., CPUPreprocessor | CUDAPreprocessor | TRTPreprocessor],
        ptype: str,
    ) -> None:
        """Preprocessors load with default settings."""
        preproc = make_preprocessor(ptype)
        assert preproc

    @pytest.mark.parametrize("ptype", ["cpu", "cuda", "trt"])
    def test_loads_with_mean_std(
        self,
        make_preprocessor: Callable[..., CPUPreprocessor | CUDAPreprocessor | TRTPreprocessor],
        ptype: str,
    ) -> None:
        """Preprocessors load with mean/std normalization."""
        preproc = make_preprocessor(ptype, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        assert preproc


class TestPreprocessorDuplicate:
    """Verify deterministic preprocessing behavior."""

    @pytest.mark.parametrize("ptype", ["cpu", "cuda", "trt"])
    def test_duplicate(
        self,
        make_preprocessor: Callable[..., CPUPreprocessor | CUDAPreprocessor | TRTPreprocessor],
        ptype: str,
        horse_image: np.ndarray,
    ) -> None:
        """Preprocessing same image yields identical results."""
        preproc = make_preprocessor(ptype)
        result1 = preproc.preprocess([horse_image])[0]
        result2 = preproc.preprocess([horse_image])[0]
        assert np.array_equal(result1, result2)

    @pytest.mark.parametrize("ptype", ["cpu", "cuda", "trt"])
    def test_duplicate_with_mean_std(
        self,
        make_preprocessor: Callable[..., CPUPreprocessor | CUDAPreprocessor | TRTPreprocessor],
        ptype: str,
        horse_image: np.ndarray,
    ) -> None:
        """Preprocessing with mean/std yields identical results."""
        preproc = make_preprocessor(ptype, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        result1 = preproc.preprocess([horse_image])[0]
        result2 = preproc.preprocess([horse_image])[0]
        assert np.array_equal(result1, result2)


class TestPreprocessorParity:
    """Check CPU/GPU preprocessor parity."""

    def _assess_parity(
        self,
        preproc1: CPUPreprocessor | CUDAPreprocessor | TRTPreprocessor,
        tag1: str,
        preproc2: CPUPreprocessor | CUDAPreprocessor | TRTPreprocessor,
        tag2: str,
        method: str,
    ) -> None:
        """Assert preprocessing outputs match across backends."""
        for img_path in IMAGE_PATHS:
            img = _read_image(img_path)
            result1, ratios1_list, padding1_list = preproc1.preprocess([img], resize=method)
            result2, ratios2_list, padding2_list = preproc2.preprocess([img], resize=method)
            ratios1, ratios2 = ratios1_list[0], ratios2_list[0]
            padding1, padding2 = padding1_list[0], padding2_list[0]
            assert ratios1 == ratios2
            assert padding1 == padding2
            assert result1.shape == result2.shape, (
                f"{tag1}: {result1.shape} != {tag2}: {result2.shape}"
            )
            assert result1.dtype == result2.dtype, (
                f"{tag1}: {result1.dtype} != {tag2}: {result2.dtype}"
            )
            cpu_mean = np.mean(result1)
            other_mean = np.mean(result2)
            assert cpu_mean - CUDA_MAG_BOUNDS <= other_mean <= cpu_mean + CUDA_MAG_BOUNDS, (
                f"{tag1}: {cpu_mean} != {tag2}: {other_mean}"
            )
            diff_mask = np.any(result1 != result2, axis=-1)
            avg_diff = np.mean(np.abs(result1[diff_mask] - result2[diff_mask]))
            assert avg_diff < 1.0, f"{tag1} != {tag2}: {avg_diff}"

    @pytest.mark.parametrize("ptype", ["cuda", "trt"])
    @pytest.mark.parametrize("method", ["linear", "letterbox"])
    def test_parity(
        self,
        make_preprocessor: Callable[..., CPUPreprocessor | CUDAPreprocessor | TRTPreprocessor],
        ptype: str,
        method: str,
    ) -> None:
        """GPU preprocessing matches CPU preprocessing."""
        cpu = make_preprocessor("cpu")
        other = make_preprocessor(ptype)
        self._assess_parity(cpu, "CPU", other, ptype.upper(), method)

    @pytest.mark.parametrize("ptype", ["cuda", "trt"])
    @pytest.mark.parametrize("method", ["linear", "letterbox"])
    def test_parity_with_mean_std(
        self,
        make_preprocessor: Callable[..., CPUPreprocessor | CUDAPreprocessor | TRTPreprocessor],
        ptype: str,
        method: str,
    ) -> None:
        """GPU preprocessing matches CPU with mean/std."""
        cpu = make_preprocessor("cpu", mean=IMAGENET_MEAN, std=IMAGENET_STD)
        other = make_preprocessor(ptype, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self._assess_parity(cpu, "CPU", other, ptype.upper(), method)


class TestBatchProcessing:
    """Validate batch preprocessing behavior."""

    @pytest.mark.parametrize("ptype", ["cpu", "cuda", "trt"])
    def test_batch_output_shape(
        self,
        make_preprocessor: Callable[..., CPUPreprocessor | CUDAPreprocessor | TRTPreprocessor],
        ptype: str,
        test_images: list[np.ndarray],
    ) -> None:
        """Batch preprocessing preserves shapes and metadata."""
        preproc = make_preprocessor(ptype)
        images = test_images[:3] if len(test_images) >= 3 else test_images
        result, ratios_list, padding_list = preproc.preprocess(images)
        assert result.shape[0] == len(images)
        assert result.shape == (len(images), 3, 640, 640)
        assert len(ratios_list) == len(images)
        assert len(padding_list) == len(images)

    @pytest.mark.parametrize("ptype", ["cuda", "trt"])
    def test_batch_parity_with_single(
        self,
        make_preprocessor: Callable[..., CPUPreprocessor | CUDAPreprocessor | TRTPreprocessor],
        ptype: str,
        test_images: list[np.ndarray],
    ) -> None:
        """Batch preprocessing matches single-image results."""
        preproc = make_preprocessor(ptype)
        images = test_images[:3] if len(test_images) >= 3 else test_images
        batch_result, batch_ratios, batch_padding = preproc.preprocess(images)
        for i, img in enumerate(images):
            single_result, single_ratios, single_padding = preproc.preprocess([img])
            assert np.allclose(batch_result[i], single_result[0], rtol=1e-5, atol=1e-5)
            assert batch_ratios[i] == single_ratios[0]
            assert batch_padding[i] == single_padding[0]

    def test_cuda_dynamic_realloc(self, horse_image: np.ndarray) -> None:
        """CUDA preprocessor reallocates for varying batch sizes."""
        preproc = CUDAPreprocessor(PREPROC_SIZE, PREPROC_RANGE, PREPROC_DTYPE)
        result1, _, _ = preproc.preprocess([horse_image])
        assert result1.shape[0] == 1
        result3, _, _ = preproc.preprocess([horse_image, horse_image, horse_image])
        assert result3.shape[0] == 3
        result2, _, _ = preproc.preprocess([horse_image, horse_image])
        assert result2.shape[0] == 2
        assert np.allclose(result1[0], result3[0], rtol=1e-5, atol=1e-5)
        assert np.allclose(result1[0], result2[0], rtol=1e-5, atol=1e-5)


class TestPerformance:
    """Benchmark preprocessing performance."""

    def _measure(
        self, images: list[np.ndarray], preproc: CPUPreprocessor | CUDAPreprocessor | TRTPreprocessor
    ) -> float:
        """
        Measure average preprocessing time.

        Returns
        -------
        float
            Average time in seconds.

        """
        profs = []
        for _ in range(10):
            t0 = time.perf_counter()
            preproc.preprocess(images)
            t1 = time.perf_counter()
            profs.append(t1 - t0)
        return float(np.mean(profs))

    def _run_perf_test(self, gpu_preproc: CUDAPreprocessor | TRTPreprocessor) -> tuple[float, float]:
        """
        Run CPU vs GPU preprocessing timing test.

        Returns
        -------
        tuple[float, float]
            CPU and GPU preprocessing times.

        """
        cpu = CPUPreprocessor(PREPROC_SIZE, PREPROC_RANGE, PREPROC_DTYPE)
        img = _read_image(HORSE_IMAGE_PATH)
        images = [img]
        for _ in range(10):
            cpu.preprocess(images)
            gpu_preproc.preprocess(images)
        cpu_time = self._measure(images, cpu)
        gpu_time = self._measure(images, gpu_preproc)
        assert cpu_time > gpu_time
        return cpu_time, gpu_time

    def test_cuda_perf(self) -> None:
        """CUDA preprocessing is faster than CPU."""
        cuda = CUDAPreprocessor(PREPROC_SIZE, PREPROC_RANGE, PREPROC_DTYPE, pagelocked_mem=False)
        cpu_time, cuda_time = self._run_perf_test(cuda)
        print(f"CPU: {cpu_time:.3f}s, CUDA: {cuda_time:.3f}s, speedup: {cpu_time / cuda_time:.2f}x")

    def test_cuda_perf_pagelocked(self) -> None:
        """CUDA preprocessing speedup with pagelocked memory."""
        cuda = CUDAPreprocessor(PREPROC_SIZE, PREPROC_RANGE, PREPROC_DTYPE, pagelocked_mem=True)
        cpu_time, cuda_time = self._run_perf_test(cuda)
        print(
            f"Pagelocked - CPU: {cpu_time:.3f}s, CUDA: {cuda_time:.3f}s, speedup: {cpu_time / cuda_time:.2f}x"
        )

    def test_trt_perf(self) -> None:
        """TRT preprocessing is faster than CPU."""
        trt = TRTPreprocessor(PREPROC_SIZE, PREPROC_RANGE, PREPROC_DTYPE, pagelocked_mem=False)
        cpu_time, trt_time = self._run_perf_test(trt)
        print(f"CPU: {cpu_time:.3f}s, TRT: {trt_time:.3f}s, speedup: {cpu_time / trt_time:.2f}x")

    def test_trt_perf_pagelocked(self) -> None:
        """TRT preprocessing speedup with pagelocked memory."""
        trt = TRTPreprocessor(PREPROC_SIZE, PREPROC_RANGE, PREPROC_DTYPE, pagelocked_mem=True)
        cpu_time, trt_time = self._run_perf_test(trt)
        print(
            f"Pagelocked - CPU: {cpu_time:.3f}s, TRT: {trt_time:.3f}s, speedup: {cpu_time / trt_time:.2f}x"
        )
