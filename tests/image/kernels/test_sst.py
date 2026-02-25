# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="misc"
"""
Consolidated tests for all SST (Scale-Swap-Transpose) CUDA kernels.

Ports from legacy:
- test_sst_kernel.py  -> SCALE_SWAP_TRANSPOSE kernel
- test_sst_fast_kernel.py -> SST_FAST / SST_FAST_F16 kernels
- test_sst_imagenet_kernel.py -> IMAGENET_SST / IMAGENET_SST_F16 kernels
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from trtutils.core import (
    Kernel,
    create_binding,
    create_stream,
    destroy_stream,
    memcpy_device_to_host_async,
    memcpy_host_to_device_async,
    stream_synchronize,
)
from trtutils.image import kernels
from trtutils.image.preprocessors import preprocess

_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
_HORSE_IMAGE_PATH = _DATA_DIR / "horse.jpg"

_KERNEL_MAP: dict[str, tuple[Any, Any]] = {
    "sst": kernels.SCALE_SWAP_TRANSPOSE,
    "sst_fast": kernels.SST_FAST,
    "sst_fast_f16": kernels.SST_FAST_F16,
    "sst_imagenet": kernels.IMAGENET_SST,
    "sst_imagenet_f16": kernels.IMAGENET_SST_F16,
}


def _run_sst_kernel(
    img: np.ndarray,
    output_height: int,
    output_width: int,
    kernel_key: str,
    batch_size: int = 1,
    scale: float = 1.0 / 255.0,
    offset: float = 0.0,
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Run an SST-family kernel and return the result."""
    kernel_data = _KERNEL_MAP[kernel_key]
    is_imagenet = kernel_key in ("sst_imagenet", "sst_imagenet_f16")
    is_f16 = kernel_key in ("sst_fast_f16", "sst_imagenet_f16")

    stream = create_stream()

    num_threads: tuple[int, int, int] = (32, 32, 1)
    num_blocks: tuple[int, int, int] = (
        math.ceil(output_width / num_threads[0]),
        math.ceil(output_height / num_threads[1]),
        batch_size,
    )

    if batch_size > 1:
        dummy_input = np.zeros((batch_size, output_height, output_width, 3), dtype=np.uint8)
        batch_img = np.stack([img] * batch_size, axis=0)
        input_data = batch_img
    else:
        dummy_input = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        input_data = img

    out_dtype = np.float16 if is_f16 else np.float32
    dummy_output = np.zeros((batch_size, 3, output_height, output_width), dtype=out_dtype)

    input_binding = create_binding(dummy_input, is_input=True)
    output_binding = create_binding(dummy_output, pagelocked_mem=True)

    kernel_obj = Kernel(kernel_data[0], kernel_data[1])

    if is_imagenet:
        assert mean is not None and std is not None
        mean_array = np.array(mean, dtype=np.float32).reshape(1, 3, 1, 1)
        std_array = np.array(std, dtype=np.float32).reshape(1, 3, 1, 1)
        mean_binding = create_binding(mean_array)
        std_binding = create_binding(std_array)

        memcpy_host_to_device_async(mean_binding.allocation, mean_array, stream)
        memcpy_host_to_device_async(std_binding.allocation, std_array, stream)

        args = kernel_obj.create_args(
            input_binding.allocation,
            output_binding.allocation,
            mean_binding.allocation,
            std_binding.allocation,
            output_height,
            output_width,
            batch_size,
        )
    else:
        args = kernel_obj.create_args(
            input_binding.allocation,
            output_binding.allocation,
            scale,
            offset,
            output_height,
            output_width,
            batch_size,
        )
        mean_binding = None
        std_binding = None

    memcpy_host_to_device_async(input_binding.allocation, input_data, stream)
    kernel_obj.call(num_blocks, num_threads, stream, args)
    memcpy_device_to_host_async(output_binding.host_allocation, output_binding.allocation, stream)
    stream_synchronize(stream)

    result = output_binding.host_allocation.copy()

    destroy_stream(stream)
    input_binding.free()
    output_binding.free()
    kernel_obj.free()
    if mean_binding is not None:
        mean_binding.free()
    if std_binding is not None:
        std_binding.free()

    return result


@pytest.mark.gpu
class TestSSTKernelCompilation:
    """Test SST kernel compilation."""

    @pytest.mark.parametrize("kernel_key", ["sst", "sst_fast", "sst_imagenet"])
    def test_compiles(self, kernel_key: str) -> None:
        """Each SST kernel variant compiles without error."""
        stream = create_stream()
        compiled = Kernel(_KERNEL_MAP[kernel_key][0], _KERNEL_MAP[kernel_key][1])
        assert compiled is not None
        destroy_stream(stream)

    @pytest.mark.parametrize(
        "kernel_key",
        ["sst_fast_f16", "sst_imagenet_f16"],
        ids=["sst_fast_f16", "sst_imagenet_f16"],
    )
    def test_f16_precision_variants_compile(self, kernel_key: str) -> None:
        """F16 precision variants compile without error."""
        stream = create_stream()
        compiled = Kernel(_KERNEL_MAP[kernel_key][0], _KERNEL_MAP[kernel_key][1])
        assert compiled is not None
        destroy_stream(stream)


@pytest.mark.gpu
class TestSSTKernelCorrectness:
    """Test SST kernel output correctness against CPU implementation."""

    @pytest.mark.parametrize("kernel_key", ["sst", "sst_fast"])
    def test_correctness_against_cpu(self, kernel_key: str) -> None:
        """GPU SST result matches CPU preprocess() output."""
        if not _HORSE_IMAGE_PATH.exists():
            pytest.skip("Horse test image not found")
        img = cv2.imread(str(_HORSE_IMAGE_PATH))
        if img is None:
            pytest.skip("Failed to read test image")

        output_height = output_width = 640
        img_resized = cv2.resize(img, (output_width, output_height))

        cuda_result = _run_sst_kernel(img_resized, output_height, output_width, kernel_key)
        cpu_result, _, _ = preprocess(
            [img_resized], (output_width, output_height), np.dtype(np.float32)
        )

        assert cuda_result.shape == cpu_result.shape
        assert np.mean(cuda_result) == np.mean(cpu_result)
        assert np.allclose(cuda_result, cpu_result)

    def test_imagenet_normalization(self) -> None:
        """SST_IMAGENET kernel applies mean/std normalization correctly."""
        if not _HORSE_IMAGE_PATH.exists():
            pytest.skip("Horse test image not found")
        img = cv2.imread(str(_HORSE_IMAGE_PATH))
        if img is None:
            pytest.skip("Failed to read test image")

        output_height = output_width = 640
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        img_resized = cv2.resize(img, (output_width, output_height))

        cuda_result = _run_sst_kernel(
            img_resized, output_height, output_width, "sst_imagenet", mean=mean, std=std
        )
        cpu_result, _, _ = preprocess(
            [img_resized],
            (output_width, output_height),
            np.dtype(np.float32),
            mean=mean,
            std=std,
        )

        assert cuda_result.shape == cpu_result.shape
        assert np.isclose(np.mean(cuda_result), np.mean(cpu_result), rtol=1e-6, atol=1e-6)
        assert np.allclose(cuda_result, cpu_result, rtol=1e-6, atol=1e-6)

    def test_sst_fast_f16_correctness(self) -> None:
        """SST_FAST_F16 kernel output matches CPU within relaxed fp16 tolerances."""
        if not _HORSE_IMAGE_PATH.exists():
            pytest.skip("Horse test image not found")
        img = cv2.imread(str(_HORSE_IMAGE_PATH))
        if img is None:
            pytest.skip("Failed to read test image")

        output_height = output_width = 640
        img_resized = cv2.resize(img, (output_width, output_height))

        cuda_result = _run_sst_kernel(img_resized, output_height, output_width, "sst_fast_f16")
        cpu_result, _, _ = preprocess(
            [img_resized], (output_width, output_height), np.dtype(np.float32)
        )

        assert cuda_result.shape == cpu_result.shape
        assert np.isclose(np.mean(cuda_result), np.mean(cpu_result), rtol=1e-3, atol=1e-3)
        assert np.allclose(cuda_result, cpu_result, rtol=1e-3, atol=1e-3)


@pytest.mark.gpu
class TestSSTBatchProcessing:
    """Test SST kernel with batch inputs."""

    @pytest.mark.parametrize("kernel_key", ["sst_fast"])
    def test_batch_matches_single(self, kernel_key: str) -> None:
        """Batch SST output matches single-image output per element."""
        if not _HORSE_IMAGE_PATH.exists():
            pytest.skip("Horse test image not found")
        img = cv2.imread(str(_HORSE_IMAGE_PATH))
        if img is None:
            pytest.skip("Failed to read test image")

        output_height = output_width = 640
        batch_size = 4
        img_resized = cv2.resize(img, (output_width, output_height))

        batch_result = _run_sst_kernel(
            img_resized, output_height, output_width, kernel_key, batch_size=batch_size
        )
        single_result = _run_sst_kernel(img_resized, output_height, output_width, kernel_key)

        for i in range(batch_size):
            assert np.allclose(batch_result[i], single_result[0])
