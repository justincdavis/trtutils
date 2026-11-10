# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/models/_model.py -- Model mixin metadata, shapes, and validation."""

from __future__ import annotations

from unittest.mock import patch

import onnx
import pytest

import trtutils.models as models
from tests.conftest import DATA_DIR
from trtutils.models import (
    RFDETR,
    DEIMv2,
    DeiT,
    DeiT3,
    PPOCRDetector,
    PPOCRRecognizer,
    RTDETRv1,
    YOLOv8,
    YOLOv10,
)
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
        pytest.param(RFDETR, "rfdetr_n", None, 384, id="rfdetr-nano"),
        pytest.param(RFDETR, "rfdetr_s", None, 512, id="rfdetr-small"),
        pytest.param(RFDETR, "rfdetr_m", None, 576, id="rfdetr-medium"),
        pytest.param(DEIMv2, "deimv2_atto", None, 320, id="variant-atto"),
        pytest.param(DEIMv2, "deimv2_femto", None, 416, id="variant-femto"),
        pytest.param(DEIMv2, "deimv2_s", None, 640, id="variant-fallback"),
        pytest.param(DEIMv2, "deimv2_atto", 320, 320, id="variant-explicit"),
        pytest.param(DeiT, "deit_base", None, 224, id="deit-default"),
        pytest.param(DeiT, "deit_base_hr", None, 384, id="deit-hr-default"),
        pytest.param(DeiT, "deit_base_hr", 448, 448, id="deit-hr-custom"),
        pytest.param(DeiT, "deit_tiny", 320, 320, id="deit-custom"),
        pytest.param(DeiT3, "deit3_huge", 252, 252, id="deit3-huge-patch14"),
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
        pytest.param(RFDETR, "rfdetr_n", 576, "requires imgsz of 384", id="rfdetr-mismatch"),
        pytest.param(DeiT, "deit3_small", None, "not supported", id="deit-other-family"),
        pytest.param(DeiT, "deit_tiny", 250, "divisible by 16", id="deit-patch"),
        pytest.param(DeiT3, "deit3_huge", 256, "divisible by 14", id="deit3-huge-patch"),
    ],
)
def test_download_rejects(tmp_path, model_cls, model, imgsz, match) -> None:
    """download() raises ValueError for unknown models and variant imgsz mismatches."""
    with pytest.raises(ValueError, match=match):
        model_cls.download(model, tmp_path / "out.onnx", imgsz=imgsz)


def test_build_unknown_kwargs_raises(tmp_path) -> None:
    """build() raises TypeError for keyword arguments no build hook accepts."""
    # yolov8's hook doesn't read the onnx, so the fake path is never opened
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        YOLOv8.build(tmp_path / "fake.onnx", tmp_path / "out.engine", totally_fake_kwarg=True)


@pytest.mark.parametrize(
    ("kwargs", "fp16"),
    [
        pytest.param({}, True, id="default"),
        pytest.param({"fp16": False}, False, id="fp16-off"),
        pytest.param({"fp16": True, "fp8": True}, True, id="fp16-with-fp8"),
        pytest.param({"strongly_typed": True}, False, id="strongly-typed"),
    ],
)
def test_build_fp16(tmp_path, kwargs, fp16) -> None:
    """build() defaults to fp16, takes an explicit fp16, and leaves strongly typed builds unflagged."""
    with patch("trtutils.models._model.build_engine") as mock_build:
        YOLOv8.build(tmp_path / "fake.onnx", tmp_path / "out.engine", **kwargs)
    call = mock_build.call_args.kwargs
    assert call["fp16"] is fp16
    for key, value in kwargs.items():
        if key != "fp16":
            assert call[key] == value


@pytest.mark.parametrize("model_cls", [PPOCRDetector, PPOCRRecognizer])
@pytest.mark.parametrize(
    ("kwargs", "fp16"),
    [
        pytest.param({}, True, id="default"),
        pytest.param({"fp16": False}, False, id="fp16-off"),
    ],
)
def test_ppocr_build_fp16(tmp_path, model_cls, kwargs, fp16) -> None:
    """The PP-OCR builds default to fp16 like the other models, and take an explicit fp16."""
    with patch("trtutils.models.ocr._ppocr.build_engine") as mock_build:
        model_cls.build(tmp_path / "fake.onnx", tmp_path / "out.engine", **kwargs)
    assert mock_build.call_args.kwargs["fp16"] is fp16


@pytest.mark.parametrize("model_cls", [PPOCRDetector, PPOCRRecognizer])
def test_ppocr_build_rejects_embed_sst(model_cls: type[Model]) -> None:
    """PP-OCR does not embed the SST, so embed_sst=True raises before any build work."""
    with patch("trtutils.models.ocr._ppocr.build_engine") as build:
        with pytest.raises(ValueError, match="embed_sst"):
            model_cls.build("model.onnx", "model.engine", embed_sst=True)
    build.assert_not_called()


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_input_tensor_names_match_onnx(model_cls) -> None:
    """_input_tensors names are the graph inputs of the exported ONNX."""
    paths = sorted((DATA_DIR / model_cls._model_type).glob("*.onnx"))
    if not paths:
        pytest.skip(f"missing {model_cls._model_type} onnx")
    graph = onnx.load(str(paths[0]), load_external_data=False).graph
    initializers = {init.name for init in graph.initializer}
    onnx_names = {i.name for i in graph.input if i.name not in initializers}
    assert {name for name, _kind in model_cls._input_tensors} == onnx_names


def _write_onnx(path, shape, name="input") -> None:
    """Write a one-node ONNX model with a single input of the given shape."""
    inp = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
    out = onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, shape)
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Relu", [name], ["out"])], "g", [inp], [out]
    )
    onnx.save(onnx.helper.make_model(graph), str(path))


def _built_shapes(model_cls, onnx_path, tmp_path, **kwargs: int):
    with patch("trtutils.models._model.build_engine") as mock_build:
        model_cls.build(onnx_path, tmp_path / "out.engine", **kwargs)
    return dict(mock_build.call_args.kwargs["shapes"])


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        pytest.param([1, 3, 384, 384], 384, id="static"),
        pytest.param([1, 3, "h", "w"], 576, id="symbolic-hw"),
        pytest.param([1, 3, 384, 320], 576, id="non-square"),
        pytest.param([1, 384, 384, 3], 576, id="nhwc"),
    ],
)
def test_build_infers_imgsz_from_onnx(tmp_path, shape, expected) -> None:
    """build() without imgsz uses the ONNX's static square H/W, else the class default."""
    path = tmp_path / "m.onnx"
    _write_onnx(path, shape)
    shapes = _built_shapes(RFDETR, path, tmp_path)
    assert shapes["input"] == (1, 3, expected, expected)


def test_build_explicit_imgsz_matching_onnx(tmp_path) -> None:
    """An explicit imgsz equal to the ONNX size builds normally."""
    path = tmp_path / "m.onnx"
    _write_onnx(path, [1, 3, 384, 384])
    assert _built_shapes(RFDETR, path, tmp_path, imgsz=384)["input"] == (1, 3, 384, 384)


def test_build_explicit_imgsz_contradicting_onnx_raises(tmp_path) -> None:
    """An explicit imgsz that differs from the static ONNX size raises before building."""
    path = tmp_path / "m.onnx"
    _write_onnx(path, [1, 3, 384, 384])
    with patch("trtutils.models._model.build_engine") as mock_build:
        with pytest.raises(ValueError, match=r"384.*576"):
            RFDETR.build(path, tmp_path / "out.engine", imgsz=576)
    mock_build.assert_not_called()


@pytest.mark.parametrize(
    ("model_cls", "rel", "expected"),
    [
        pytest.param(RFDETR, "rfdetr/rfdetr_n_384.onnx", 384, id="rfdetr_n"),
        pytest.param(RFDETR, "rfdetr/rfdetr_s_512.onnx", 512, id="rfdetr_s"),
        pytest.param(DEIMv2, "deimv2/deimv2_atto_320.onnx", 320, id="deimv2_atto"),
        pytest.param(DEIMv2, "deimv2/deimv2_femto_416.onnx", 416, id="deimv2_femto"),
    ],
)
def test_build_infers_imgsz_real_onnx(tmp_path, model_cls, rel, expected) -> None:
    """build() without imgsz picks the fixed size of the real per-variant exports."""
    path = DATA_DIR / rel
    if not path.exists():
        pytest.skip(f"{path} missing")
    shapes = _built_shapes(model_cls, path, tmp_path)
    image_name = next(n for n, kind in model_cls._input_tensors if kind == "image")
    assert shapes[image_name] == (1, 3, expected, expected)
