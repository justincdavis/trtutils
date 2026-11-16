# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/_image_model.py -- ImageModel preprocessing and engine helpers."""

from __future__ import annotations

from unittest import mock

import numpy as np
import nvtx
import pytest

from trtutils._flags import FLAGS
from trtutils._log import LOG
from trtutils.image import Detector, ImageModel
from trtutils.image.preprocessors import CPUPreprocessor, CUDAPreprocessor, HybridPreprocessor


@pytest.mark.parametrize("preprocessor", ["cpu", "cuda", "trt", "hybrid"])
@pytest.mark.parametrize("resize_method", ["linear", "letterbox"])
def test_preprocess(yolov10_engine, images, preprocessor, resize_method) -> None:
    """preprocess() yields a (1, 3, H, W) tensor in the model dtype for every backend and resize."""
    model = ImageModel(
        yolov10_engine, preprocessor=preprocessor, resize_method=resize_method, warmup=False
    )
    tensor, ratios, padding = model.preprocess(images["horse"].array)
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == model.dtype
    assert len(ratios) == 1
    assert len(padding) == 1


@pytest.mark.parametrize(
    ("preprocessor", "preproc_cls", "atol"),
    [
        # the cuda resize kernels differ from cv2 by at most a pixel level
        pytest.param("cuda", CUDAPreprocessor, 2 / 255, id="cuda"),
        pytest.param("hybrid", HybridPreprocessor, 0.0, id="hybrid"),
    ],
)
def test_fp16_input_gpu_preprocessor_matches_cpu(
    yolov10_fp16_engine, images, preprocessor, preproc_cls, atol
) -> None:
    """A float16-input model keeps its GPU preprocessor, whose tensor and detections match the CPU's."""
    model = Detector(yolov10_fp16_engine, preprocessor=preprocessor, warmup=False)
    cpu = Detector(yolov10_fp16_engine, preprocessor="cpu", warmup=False)
    assert type(model.get_preprocessor()) is preproc_cls
    image = images["horse"].array

    tensor, ratios, padding = model.preprocess(image)
    expected, cpu_ratios, cpu_padding = cpu.preprocess(image)
    assert tensor.dtype == expected.dtype == np.float16
    assert (ratios, padding) == (cpu_ratios, cpu_padding)
    np.testing.assert_allclose(tensor, expected, rtol=0, atol=atol)

    bboxes, scores, class_ids = model.run(image)
    cpu_bboxes, cpu_scores, cpu_class_ids = cpu.run(image)
    np.testing.assert_allclose(bboxes, cpu_bboxes, atol=1.0)
    np.testing.assert_allclose(scores, cpu_scores, atol=0.01)
    np.testing.assert_array_equal(class_ids, cpu_class_ids)


@pytest.mark.parametrize("preprocessor", ["cuda", "hybrid"])
def test_non_float_input_falls_back_to_cpu(uint8_nchw_engine, preprocessor) -> None:
    """A uint8 NCHW input, which the GPU preprocessors cannot write, warns and uses the CPU."""
    with mock.patch.object(LOG, "warning", wraps=LOG.warning) as warning:
        model = ImageModel(
            uint8_nchw_engine, input_range=(0.0, 255.0), preprocessor=preprocessor, warmup=False
        )
    assert type(model.get_preprocessor()) is CPUPreprocessor
    assert any("fallback to CPU" in call.args[0] for call in warning.call_args_list)


def test_get_random_input_and_mock_run(yolov10_engine) -> None:
    """get_random_input() matches the engine input spec and mock_run() matches the output spec."""
    model = ImageModel(yolov10_engine, warmup=False)
    rand_input = model.get_random_input()
    assert [(list(a.shape), a.dtype) for a in rand_input] == model.engine.input_spec
    outputs = model.mock_run()
    assert [(list(o.shape), o.dtype) for o in outputs] == model.engine.output_spec


def test_init_failure_balances_nvtx_ranges(yolov10_engine, monkeypatch) -> None:
    """A constructor that raises still pops every NVTX range it pushed."""
    depth = []
    monkeypatch.setattr(FLAGS, "NVTX_ENABLED", True)
    monkeypatch.setattr(nvtx, "push_range", depth.append)
    monkeypatch.setattr(nvtx, "pop_range", depth.pop)
    monkeypatch.setattr(ImageModel, "_configure_model", mock.Mock(side_effect=ValueError("bad")))
    with pytest.raises(ValueError, match="bad"):
        ImageModel(yolov10_engine, warmup=False)
    assert depth == []
