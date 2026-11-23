# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Engine-level tests for the image restorer wrappers in src/trtutils/models/image_restorers/."""

from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort
import pytest

from tests.conftest import DATA_DIR, PREPROCESSORS
from trtutils import Buffer, MemoryLocation
from trtutils.image._schema import ImageRestorerOutputSchema
from trtutils.models import RealESRGAN

_HEIGHT, _WIDTH = 480, 640

IMAGE_RESTORERS = [
    pytest.param(
        DATA_DIR / "realesrgan" / f"realesrgan_general_x4v3_{_HEIGHT}x{_WIDTH}.onnx",
        4,
        id="realesrgan-general-x4v3",
    ),
    pytest.param(
        DATA_DIR / "realesrgan" / f"realesrgan_x2plus_{_HEIGHT}x{_WIDTH}.onnx",
        2,
        id="realesrgan-x2plus",
    ),
    pytest.param(
        DATA_DIR / "realesrgan" / f"realesrgan_x4plus_{_HEIGHT}x{_WIDTH}.onnx",
        4,
        id="realesrgan-x4plus",
    ),
]


def _street_crop() -> np.ndarray:
    """A 480x640 crop of data/street.jpg, the engine input size."""
    image = cv2.imread(str(DATA_DIR / "street.jpg"))
    top, left = (image.shape[0] - _HEIGHT) // 2, (image.shape[1] - _WIDTH) // 2
    return np.ascontiguousarray(image[top : top + _HEIGHT, left : left + _WIDTH])


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse))


@pytest.mark.parametrize(("onnx_path", "scale"), IMAGE_RESTORERS)
@pytest.mark.parametrize(("preprocessor", "cpu_resize"), PREPROCESSORS)
def test_image_restorer_end2end(
    build_model_engine, images, onnx_path, scale, preprocessor, cpu_resize
) -> None:
    """run(), get_restored_images() and end2end() agree on padded and exact-size images."""
    engine = build_model_engine(RealESRGAN, onnx_path)
    model = RealESRGAN(engine, preprocessor=preprocessor, cpu_resize=cpu_resize, warmup=False)
    reference = RealESRGAN(engine, preprocessor="cpu", warmup=False)
    assert model.output_schema is ImageRestorerOutputSchema.REAL_ESRGAN
    assert model.scale == scale
    # horse is 288x447, padded to the 480x640 input
    small = images["horse"].array
    exact = _street_crop()

    restored = []
    for image in (small, exact):
        via_run = model.get_restored_images(model.run(image))
        via_e2e = model.end2end(image)
        np.testing.assert_array_equal(via_run, via_e2e)
        assert via_e2e.shape == (scale * image.shape[0], scale * image.shape[1], 3)
        assert via_e2e.dtype == np.uint8
        # every preprocessor feeds the engine the same padded image
        diff = np.abs(via_e2e.astype(int) - reference.end2end(image))
        assert diff.max() <= 1
        restored.append((via_e2e, via_e2e.copy()))
    # end2end returns new arrays: later calls did not overwrite the earlier results
    for result, kept in restored:
        np.testing.assert_array_equal(result, kept)


@pytest.mark.parametrize(("onnx_path", "scale"), IMAGE_RESTORERS)
def test_image_restorer_inputs(build_model_engine, images, onnx_path, scale) -> None:
    """Device Buffers, preprocessed batches and sizes give the same image; larger images raise."""
    engine = build_model_engine(RealESRGAN, onnx_path)
    model = RealESRGAN(engine, warmup=False)
    small = images["horse"].array
    expected = model.end2end(small)

    np.testing.assert_array_equal(
        model.end2end(Buffer.from_array(small, MemoryLocation.DEVICE)), expected
    )
    tensor, _, _ = model.preprocess(small)
    # a preprocessed batch tensor gives batch results
    outputs = model.run([tensor], small.shape[:2], preprocessed=True)
    np.testing.assert_array_equal(model.get_restored_images(outputs)[0], expected)
    # without sizes a preprocessed batch keeps the whole padded output
    whole = model.get_restored_images(model.run([tensor], preprocessed=True))[0]
    assert whole.shape == (scale * _HEIGHT, scale * _WIDTH, 3)
    np.testing.assert_array_equal(whole[: expected.shape[0], : expected.shape[1]], expected)

    with pytest.raises(ValueError, match="build the engine at the image size"):
        model.end2end(np.zeros((_HEIGHT + 1, _WIDTH, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="does not rescale"):
        model.preprocess(small, resize="linear")


@pytest.mark.parametrize(("onnx_path", "scale"), IMAGE_RESTORERS)
def test_image_restorer_cuda_postprocessor_matches_cpu(
    build_model_engine, images, onnx_path, scale
) -> None:
    """The default CUDA postprocessor gives the CPU images exactly, via run() and end2end()."""
    engine = build_model_engine(RealESRGAN, onnx_path)
    cpu = RealESRGAN(engine, postprocessor="cpu", warmup=False)
    cuda = RealESRGAN(engine, warmup=False)
    assert cuda._postprocessor == "cuda"

    for image in (images["horse"].array, _street_crop()):
        expected = cpu.end2end(image)
        assert expected.shape == (scale * image.shape[0], scale * image.shape[1], 3)
        first = cuda.end2end(image)
        for restored in (first, cuda.get_restored_images(cuda.run(image)), cuda.end2end(image)):
            np.testing.assert_array_equal(restored, expected)
        # the later calls reused the mapped buffer, not the returned array
        np.testing.assert_array_equal(first, expected)


@pytest.mark.correctness
@pytest.mark.parametrize(("onnx_path", "scale"), IMAGE_RESTORERS)
def test_image_restorer_matches_onnxruntime(build_model_engine, onnx_path, scale) -> None:
    """The default (FP16) engine restores a real image within 40 dB PSNR of onnxruntime FP32."""
    engine = build_model_engine(RealESRGAN, onnx_path)
    model = RealESRGAN(engine, warmup=False)
    image = _street_crop()
    restored = model.end2end(image)

    tensor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None] / np.float32(255)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = session.run(None, {"input": tensor.astype(np.float32)})[0][0]
    # Real-ESRGAN's own conversion to a uint8 BGR image
    reference = (np.clip(out, 0, 1)[[2, 1, 0]].transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    assert restored.shape == reference.shape == (scale * _HEIGHT, scale * _WIDTH, 3)
    assert _psnr(restored, reference) > 40.0
