# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import time

import cv2
import numpy as np

from trtutils.image.preprocessors import (
    CPUPreprocessor,
    CUDAPreprocessor,
    TRTPreprocessor,
)

from .paths import HORSE_IMAGE_PATH, IMAGE_PATHS

CUDA_MAG_BOUNDS = 0.01
TRT_MAG_BOUNDS = 0.01


def _read_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        err_msg = f"Failed to read image: {path}"
        raise FileNotFoundError(err_msg)
    return img


def test_cpu_preproc_loads() -> None:
    """Test if the CPUPreprocessor loads."""
    preproc = CPUPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))
    assert preproc


def test_cpu_preproc_loads_with_mean_std() -> None:
    """Test if the CPUPreprocessor loads with mean and std."""
    preproc = CPUPreprocessor(
        (640, 640),
        (0.0, 1.0),
        np.dtype(np.float32),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    assert preproc


def test_cuda_preproc_loads() -> None:
    """Test if the CPUPreprocessor loads."""
    preproc = CUDAPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))
    assert preproc


def test_cuda_preproc_loads_with_mean_std() -> None:
    """Test if the CUDAPreprocessor loads with mean and std."""
    preproc = CUDAPreprocessor(
        (640, 640),
        (0.0, 1.0),
        np.dtype(np.float32),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    assert preproc


def test_trt_preproc_loads() -> None:
    """Test if the CPUPreprocessor loads."""
    preproc = TRTPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))
    assert preproc


def test_trt_preproc_loads_with_mean_std() -> None:
    """Test if the TRTPreprocessor loads with mean and std."""
    preproc = TRTPreprocessor(
        (640, 640),
        (0.0, 1.0),
        np.dtype(np.float32),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    assert preproc


def test_cpu_preproc_duplicate() -> None:
    """Checks that the same data will give same results with CPU."""
    preproc = CPUPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))
    img = _read_image(HORSE_IMAGE_PATH)
    result1 = preproc.preprocess([img])[0]
    result2 = preproc.preprocess([img])[0]
    assert np.array_equal(result1, result2)


def test_cpu_preproc_duplicate_with_mean_std() -> None:
    """Checks that the same data will give same results with CPU."""
    preproc = CPUPreprocessor(
        (640, 640),
        (0.0, 1.0),
        np.dtype(np.float32),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    img = _read_image(HORSE_IMAGE_PATH)
    result1 = preproc.preprocess([img])[0]
    result2 = preproc.preprocess([img])[0]
    assert np.array_equal(result1, result2)


def test_cuda_preproc_duplicate() -> None:
    """Checks that the same data will give same results with CUDA."""
    preproc = CUDAPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))
    img = _read_image(HORSE_IMAGE_PATH)
    result1 = preproc.preprocess([img])[0]
    result2 = preproc.preprocess([img])[0]
    assert np.array_equal(result1, result2)


def test_cuda_preproc_duplicate_with_mean_std() -> None:
    """Checks that the same data will give same results with CUDA."""
    preproc = CUDAPreprocessor(
        (640, 640),
        (0.0, 1.0),
        np.dtype(np.float32),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    img = _read_image(HORSE_IMAGE_PATH)
    result1 = preproc.preprocess([img])[0]
    result2 = preproc.preprocess([img])[0]
    assert np.array_equal(result1, result2)


def test_trt_preproc_duplicate() -> None:
    """Checks that the same data will give same results with TRT."""
    preproc = TRTPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))
    img = _read_image(HORSE_IMAGE_PATH)
    result1 = preproc.preprocess([img])[0]
    result2 = preproc.preprocess([img])[0]
    assert np.array_equal(result1, result2)


def test_trt_preproc_duplicate_with_mean_std() -> None:
    """Checks that the same data will give same results with TRT."""
    preproc = TRTPreprocessor(
        (640, 640),
        (0.0, 1.0),
        np.dtype(np.float32),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    img = _read_image(HORSE_IMAGE_PATH)
    result1 = preproc.preprocess([img])[0]
    result2 = preproc.preprocess([img])[0]
    assert np.array_equal(result1, result2)


def _assess_parity(
    preproc1: CPUPreprocessor | CUDAPreprocessor | TRTPreprocessor,
    tag1: str,
    preproc2: CPUPreprocessor | CUDAPreprocessor | TRTPreprocessor,
    tag2: str,
    method: str,
) -> None:
    for img_path in IMAGE_PATHS:
        img = _read_image(img_path)
        result1, ratios1_list, padding1_list = preproc1.preprocess([img], resize=method)
        result2, ratios2_list, padding2_list = preproc2.preprocess([img], resize=method)
        # Extract first element since we passed single image
        ratios1, ratios2 = ratios1_list[0], ratios2_list[0]
        padding1, padding2 = padding1_list[0], padding2_list[0]
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


def test_cuda_parity_linear_with_mean_std() -> None:
    """Test the results of the CUDA preprocessor againist the CPU preprocessor."""
    cpu = CPUPreprocessor(
        (640, 640),
        (0.0, 1.0),
        np.dtype(np.float32),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    cuda = CUDAPreprocessor(
        (640, 640),
        (0.0, 1.0),
        np.dtype(np.float32),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    _assess_parity(cpu, "CPU", cuda, "CUDA", "linear")


def test_cuda_parity_letterbox_with_mean_std() -> None:
    """Test the results of the CUDA preprocessor againist the CPU preprocessor."""
    cpu = CPUPreprocessor(
        (640, 640),
        (0.0, 1.0),
        np.dtype(np.float32),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    cuda = CUDAPreprocessor(
        (640, 640),
        (0.0, 1.0),
        np.dtype(np.float32),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

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


def test_trt_parity_linear_with_mean_std() -> None:
    """Test the results of the TRT preprocessor againist the CPU preprocessor."""
    cpu = CPUPreprocessor(
        (640, 640),
        (0.0, 1.0),
        np.dtype(np.float32),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    trt = TRTPreprocessor(
        (640, 640),
        (0.0, 1.0),
        np.dtype(np.float32),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    _assess_parity(cpu, "CPU", trt, "TRT", "linear")


def test_trt_parity_letterbox_with_mean_std() -> None:
    """Test the results of the TRT preprocessor againist the CPU preprocessor."""
    cpu = CPUPreprocessor(
        (640, 640),
        (0.0, 1.0),
        np.dtype(np.float32),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    trt = TRTPreprocessor(
        (640, 640),
        (0.0, 1.0),
        np.dtype(np.float32),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    _assess_parity(cpu, "CPU", trt, "TRT", "letterbox")


def _measure(
    images: list[np.ndarray], preproc: CPUPreprocessor | CUDAPreprocessor | TRTPreprocessor
) -> float:
    profs = []
    for _ in range(10):
        t0 = time.perf_counter()
        preproc.preprocess(images)
        t1 = time.perf_counter()
        profs.append(t1 - t0)
    return float(np.mean(profs))


def _cuda_perf(*, pagelocked_mem: bool) -> tuple[float, float]:
    cpu = CPUPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))
    cuda = CUDAPreprocessor(
        (640, 640), (0.0, 1.0), np.dtype(np.float32), pagelocked_mem=pagelocked_mem
    )
    img = _read_image(HORSE_IMAGE_PATH)
    images = [img]
    for _ in range(10):
        cpu.preprocess(images)
        cuda.preprocess(images)
    return _measure(images, cpu), _measure(images, cuda)


def test_cuda_perf() -> None:
    """Test that the CUDA preprocessor is faster than the CPU preprocessor."""
    cpu_time, cuda_time = _cuda_perf(pagelocked_mem=False)
    assert cpu_time > cuda_time
    print(
        f"CPU time: {cpu_time:.3f}s, CUDA time: {cuda_time:.3f}s, speed up: {cpu_time / cuda_time:.2f}x"
    )


def test_cuda_perf_pagelocked() -> None:
    """Test that the CUDA preprocessor is faster than the CPU preprocessor."""
    cpu_time, cuda_time = _cuda_perf(pagelocked_mem=True)
    assert cpu_time > cuda_time
    print(
        f"Pagelocked - CPU time: {cpu_time:.3f}s, CUDA time: {cuda_time:.3f}s, speed up: {cpu_time / cuda_time:.2f}x"
    )


def _trt_perf(*, pagelocked_mem: bool) -> tuple[float, float]:
    cpu = CPUPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))
    trt = TRTPreprocessor(
        (640, 640), (0.0, 1.0), np.dtype(np.float32), pagelocked_mem=pagelocked_mem
    )
    img = _read_image(HORSE_IMAGE_PATH)
    images = [img]
    for _ in range(10):
        cpu.preprocess(images)
        trt.preprocess(images)
    return _measure(images, cpu), _measure(images, trt)


def test_trt_perf() -> None:
    """Test that the TRT preprocessor is faster than the CPU preprocessor."""
    cpu_time, trt_time = _trt_perf(pagelocked_mem=False)
    assert cpu_time > trt_time
    print(
        f"CPU time: {cpu_time:.3f}s, TRT time: {trt_time:.3f}s, speed up: {cpu_time / trt_time:.2f}x"
    )


def test_trt_perf_pagelocked() -> None:
    """Test that the TRT preprocessor is faster than the CPU preprocessor."""
    cpu_time, trt_time = _trt_perf(pagelocked_mem=True)
    assert cpu_time > trt_time
    print(
        f"Pagelocked - CPU time: {cpu_time:.3f}s, TRT time: {trt_time:.3f}s, speed up: {cpu_time / trt_time:.2f}x"
    )


# ============ Batch Tests ============


def test_cpu_batch_output_shape() -> None:
    """Test that batch preprocessing returns correct output shape."""
    preproc = CPUPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))
    images = [_read_image(p) for p in IMAGE_PATHS[:3]]
    result, ratios_list, padding_list = preproc.preprocess(images)
    assert result.shape[0] == len(images)
    assert result.shape == (len(images), 3, 640, 640)
    assert len(ratios_list) == len(images)
    assert len(padding_list) == len(images)


def test_cuda_batch_output_shape() -> None:
    """Test that batch preprocessing returns correct output shape."""
    preproc = CUDAPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))
    images = [_read_image(p) for p in IMAGE_PATHS[:3]]
    result, ratios_list, padding_list = preproc.preprocess(images)
    assert result.shape[0] == len(images)
    assert result.shape == (len(images), 3, 640, 640)
    assert len(ratios_list) == len(images)
    assert len(padding_list) == len(images)


def test_trt_batch_output_shape() -> None:
    """Test that batch preprocessing returns correct output shape."""
    preproc = TRTPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32), batch_size=4)
    images = [_read_image(p) for p in IMAGE_PATHS[:3]]
    result, ratios_list, padding_list = preproc.preprocess(images)
    assert result.shape[0] == len(images)
    assert result.shape == (len(images), 3, 640, 640)
    assert len(ratios_list) == len(images)
    assert len(padding_list) == len(images)


def test_cuda_batch_parity_with_single() -> None:
    """Test that batch results match single-image results."""
    preproc = CUDAPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))
    images = [_read_image(p) for p in IMAGE_PATHS[:3]]

    # Process as batch
    batch_result, batch_ratios, batch_padding = preproc.preprocess(images)

    # Process individually
    for i, img in enumerate(images):
        single_result, single_ratios, single_padding = preproc.preprocess([img])
        assert np.allclose(batch_result[i], single_result[0], rtol=1e-5, atol=1e-5)
        assert batch_ratios[i] == single_ratios[0]
        assert batch_padding[i] == single_padding[0]


def test_trt_batch_parity_with_single() -> None:
    """Test that batch results match single-image results."""
    preproc = TRTPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32), batch_size=4)
    images = [_read_image(p) for p in IMAGE_PATHS[:3]]

    # Process as batch
    batch_result, batch_ratios, batch_padding = preproc.preprocess(images)

    # Process individually
    for i, img in enumerate(images):
        single_result, single_ratios, single_padding = preproc.preprocess([img])
        assert np.allclose(batch_result[i], single_result[0], rtol=1e-5, atol=1e-5)
        assert batch_ratios[i] == single_ratios[0]
        assert batch_padding[i] == single_padding[0]


def test_cuda_batch_dynamic_realloc() -> None:
    """Test that CUDA preprocessor correctly reallocates for different batch sizes."""
    preproc = CUDAPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))
    img = _read_image(HORSE_IMAGE_PATH)

    # Process with batch size 1
    result1, _, _ = preproc.preprocess([img])
    assert result1.shape[0] == 1

    # Process with batch size 3
    result3, _, _ = preproc.preprocess([img, img, img])
    assert result3.shape[0] == 3

    # Process with batch size 2
    result2, _, _ = preproc.preprocess([img, img])
    assert result2.shape[0] == 2

    # Verify results are consistent
    assert np.allclose(result1[0], result3[0], rtol=1e-5, atol=1e-5)
    assert np.allclose(result1[0], result2[0], rtol=1e-5, atol=1e-5)
