# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/models/_model.py -- Model mixin metadata, shapes, and validation."""

from __future__ import annotations

from unittest.mock import patch

import pytest

import trtutils.models as models
from trtutils.models import RFDETR, DEIMv2, RTDETRv1, YOLOv10
from trtutils.models._model import Model

pytestmark = pytest.mark.cpu

MODEL_CLASSES = [
    pytest.param(getattr(models, name), id=name)
    for name in models.__all__
    if issubclass(getattr(models, name), Model)
]


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_class_attributes_and_shapes(model_cls) -> None:
    """Every Model subclass declares its metadata and _make_shapes mirrors _input_tensors."""
    assert isinstance(model_cls._model_type, str)
    assert model_cls._model_type
    assert isinstance(model_cls._friendly_name, str)
    assert model_cls._friendly_name
    assert isinstance(model_cls._default_imgsz, int)
    assert model_cls._default_imgsz > 0
    assert len(model_cls._input_tensors) > 0

    imgsz = model_cls._default_imgsz
    shapes = model_cls._make_shapes(4, imgsz)
    assert len(shapes) == len(model_cls._input_tensors)
    for (name, kind), (shape_name, shape) in zip(model_cls._input_tensors, shapes):
        assert shape_name == name
        assert kind in ("image", "size")
        if kind == "image":
            assert shape == (4, 3, imgsz, imgsz)
        else:
            assert shape == (4, 2)


def test_make_shapes_unknown_kind_raises() -> None:
    """_make_shapes raises ValueError for an unknown input tensor kind."""

    class FakeModel(Model):
        _model_type = "fake"
        _friendly_name = "Fake"
        _default_imgsz = 640
        _input_tensors = [("input", "unknown")]  # noqa: RUF012

    with pytest.raises(ValueError, match="Unknown input tensor kind"):
        FakeModel._make_shapes(1, 640)


@pytest.mark.parametrize(
    ("model_cls", "imgsz"),
    [
        pytest.param(RTDETRv1, 640, id="valid-list"),
        pytest.param(RFDETR, 576, id="divisor"),
        pytest.param(YOLOv10, 123, id="unrestricted"),
    ],
)
def test_validate_imgsz_accepts(model_cls, imgsz) -> None:
    """_validate_imgsz accepts sizes allowed by _valid_imgszs and _imgsz_divisor."""
    model_cls._validate_imgsz(imgsz)


@pytest.mark.parametrize(
    ("model_cls", "imgsz", "match"),
    [
        pytest.param(RTDETRv1, 320, "supports only imgsz", id="valid-list"),
        pytest.param(RFDETR, 577, "divisible by", id="divisor"),
    ],
)
def test_validate_imgsz_rejects(model_cls, imgsz, match) -> None:
    """_validate_imgsz raises ValueError for sizes outside _valid_imgszs or off _imgsz_divisor."""
    with pytest.raises(ValueError, match=match):
        model_cls._validate_imgsz(imgsz)


@pytest.mark.parametrize(
    ("model_cls", "model", "imgsz", "expected"),
    [
        pytest.param(YOLOv10, "yolov10n", None, 640, id="default"),
        pytest.param(RFDETR, "rfdetr_n", None, 576, id="default-rfdetr"),
        pytest.param(DEIMv2, "deimv2_atto", None, 320, id="variant-atto"),
        pytest.param(DEIMv2, "deimv2_femto", None, 416, id="variant-femto"),
        pytest.param(DEIMv2, "deimv2_s", None, 640, id="variant-fallback"),
        pytest.param(DEIMv2, "deimv2_atto", 320, 320, id="variant-explicit"),
    ],
)
def test_download_resolves_imgsz(tmp_path, model_cls, model, imgsz, expected) -> None:
    """download() forwards the class default or variant-specific imgsz when none is given."""
    with patch("trtutils.models._model.download_model_internal") as mock_download:
        model_cls.download(model, tmp_path / "out.onnx", imgsz=imgsz)
    assert mock_download.call_args.kwargs["imgsz"] == expected


@pytest.mark.parametrize(
    ("model_cls", "model", "imgsz", "match"),
    [
        pytest.param(YOLOv10, "fake_model_xyz", None, "not supported", id="unknown-model"),
        pytest.param(DEIMv2, "deimv2_atto", 640, "requires imgsz of 320", id="variant-mismatch"),
    ],
)
def test_download_rejects(tmp_path, model_cls, model, imgsz, match) -> None:
    """download() raises ValueError for unknown models and variant imgsz mismatches."""
    with pytest.raises(ValueError, match=match):
        model_cls.download(model, tmp_path / "out.onnx", imgsz=imgsz)


def test_build_unknown_kwargs_raises(tmp_path) -> None:
    """build() raises TypeError for keyword arguments no build hook accepts."""
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        YOLOv10.build(tmp_path / "fake.onnx", tmp_path / "out.engine", totally_fake_kwarg=True)
