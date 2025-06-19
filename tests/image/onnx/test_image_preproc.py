# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from trtutils import TRTEngine, set_log_level
from trtutils.image.onnx_models import build_image_preproc
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

    # cpu version
    cpu_result, _, _ = preprocess(
        img, (output_shape, output_shape), np.dtype(np.float32)
    )

    # trt version
    engine_path = build_image_preproc(
        (output_shape, output_shape), np.dtype(np.float32)
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

    # compare
    assert trt_result.shape == cpu_result.shape
    assert trt_result.dtype == cpu_result.dtype
    assert np.min(trt_result) >= 0.0
    assert np.max(trt_result) <= 1.0
    cpu_mean = np.mean(cpu_result)
    trt_mean = np.mean(trt_result)
    assert cpu_mean * 0.99 <= trt_mean <= cpu_mean * 1.01, (
        f"CPU: {cpu_mean}, TRT: {trt_mean}"
    )
    assert np.min(trt_result) == np.min(cpu_result)
    assert np.max(trt_result) == np.max(cpu_result)

    diff_mask = np.any(cpu_result != trt_result, axis=-1)
    avg_diff = np.mean(np.abs(cpu_result[diff_mask] - trt_result[diff_mask]))
    num_pixels = np.sum(diff_mask)
    print(f"Num pixels: {num_pixels}, avg diff: {avg_diff}")

    # use small tolerances since the TRT engine uses fp16, and CPU uses fp32
    assert avg_diff < 0.0001, f"Num pixels: {num_pixels}, avg diff: {avg_diff}"
    assert np.allclose(trt_result, cpu_result, rtol=5e-4, atol=5e-4)

    del engine


if __name__ == "__main__":
    set_log_level("DEBUG")
    test_trt_preproc_engine()
