# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/_image_model.py -- ImageModel preprocessing and engine helpers."""

from __future__ import annotations

from unittest import mock

import nvtx
import pytest

from trtutils._flags import FLAGS
from trtutils.image import ImageModel


@pytest.mark.parametrize("preprocessor", ["cpu", "cuda", "trt"])
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
