# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib
from pathlib import Path

import cv2
import numpy as np

with contextlib.suppress(ImportError):
    import tensorrt as trt

from trtutils import TRTEngine, set_log_level
from trtutils.image.onnx_models import build_image_preproc, build_image_preproc_imagenet
from trtutils.image.preprocessors import preprocess

IMG_PATH = str(Path(__file__).parent.parent.parent.parent / "data" / "horse.jpg")


def test_trt_preproc_engine() -> None:
    """Test TRT preprocessing engine as standalone kernel."""
    output_shape = 640
    o_range = (0.0, 1.0)
    scale = o_range[1] / 255.0
    offset = o_range[0]

    # resize for both methods
    img = cv2.imread(IMG_PATH)
    img = cv2.resize(img, (output_shape, output_shape))

    # cpu version (now expects list of images)
    cpu_result, _, _ = preprocess(
        [img], (output_shape, output_shape), np.dtype(np.float32), input_range=o_range
    )
    # Extract first image from batch result
    cpu_result = cpu_result[0]

    # trt version
    engine_path = build_image_preproc(
        (output_shape, output_shape),
        np.dtype(np.float32),
        trt_version=trt.__version__,
    )
    engine = TRTEngine(engine_path)
    engine.mock_execute()

    all_result = engine.execute(
        [
            img,
            np.array((scale,), dtype=np.float32),
            np.array((offset,), dtype=np.float32),
        ]
    )
    trt_result = all_result[0]  # only one output
    # Extract first image from batch (if batched)
    if trt_result.ndim == 4:
        trt_result = trt_result[0]

    # compare
    assert trt_result.shape == cpu_result.shape
    assert trt_result.dtype == cpu_result.dtype
    assert np.min(trt_result) >= 0.0
    assert np.max(trt_result) <= 1.0
    cpu_mean = np.mean(cpu_result)
    trt_mean = np.mean(trt_result)
    assert cpu_mean * 0.99 <= trt_mean <= cpu_mean * 1.01, f"CPU: {cpu_mean}, TRT: {trt_mean}"
    assert np.min(trt_result) == np.min(cpu_result)  # type: ignore[operator]
    assert np.max(trt_result) == np.max(cpu_result)  # type: ignore[operator]

    diff_mask = np.any(cpu_result != trt_result, axis=-1)
    avg_diff = np.mean(np.abs(cpu_result[diff_mask] - trt_result[diff_mask]))
    num_pixels: float = np.sum(diff_mask)
    print(f"Num pixels: {num_pixels}, avg diff: {avg_diff}")

    # use small tolerances since the TRT engine uses fp16, and CPU uses fp32
    assert avg_diff < 0.0001, f"Num pixels: {num_pixels}, avg diff: {avg_diff}"
    assert np.allclose(trt_result, cpu_result, rtol=5e-4, atol=5e-4)

    del engine


def test_trt_preproc_imagenet_engine() -> None:
    """Test TRT ImageNet preprocessing engine as standalone kernel."""
    output_shape = 640
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # resize for both methods
    img = cv2.imread(IMG_PATH)
    img = cv2.resize(img, (output_shape, output_shape))

    # cpu version with mean/std (now expects list of images)
    cpu_result, _, _ = preprocess(
        [img],
        (output_shape, output_shape),
        np.dtype(np.float32),
        input_range=(0.0, 1.0),
        mean=mean,
        std=std,
    )
    # Extract first image from batch result
    cpu_result = cpu_result[0]

    # trt version
    engine_path = build_image_preproc_imagenet(
        (output_shape, output_shape),
        np.dtype(np.float32),
        trt_version=trt.__version__,
    )
    engine = TRTEngine(engine_path)
    engine.mock_execute()

    # Format mean and std as (1, 3, 1, 1) arrays
    mean_array = np.array(mean, dtype=np.float32).reshape(1, 3, 1, 1)
    std_array = np.array(std, dtype=np.float32).reshape(1, 3, 1, 1)

    all_result = engine.execute([img, mean_array, std_array])
    trt_result = all_result[0]  # only one output
    # Extract first image from batch (if batched)
    if trt_result.ndim == 4:
        trt_result = trt_result[0]

    # compare
    assert trt_result.shape == cpu_result.shape
    assert trt_result.dtype == cpu_result.dtype
    cpu_mean = np.mean(cpu_result)
    trt_mean = np.mean(trt_result)
    assert cpu_mean * 0.99 <= trt_mean <= cpu_mean * 1.01, f"CPU: {cpu_mean}, TRT: {trt_mean}"

    diff_mask = np.any(cpu_result != trt_result, axis=-1)
    avg_diff = np.mean(np.abs(cpu_result[diff_mask] - trt_result[diff_mask]))
    num_pixels: float = np.sum(diff_mask)
    print(f"Num pixels: {num_pixels}, avg diff: {avg_diff}")

    # use small tolerances since the TRT engine uses fp16, and CPU uses fp32
    assert avg_diff < 0.0001, f"Num pixels: {num_pixels}, avg diff: {avg_diff}"
    assert np.allclose(trt_result, cpu_result, rtol=5e-4, atol=5e-4)

    del engine


if __name__ == "__main__":
    set_log_level("DEBUG")
    test_trt_preproc_engine()
    test_trt_preproc_imagenet_engine()
