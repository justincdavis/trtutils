# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/models/_model.py -- Model mixin metadata, shapes, and validation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import onnx
import pytest

import trtutils.models as models
from tests.conftest import DATA_DIR
from trtutils._flags import FLAGS
from trtutils.models import (
    RFDETR,
    DEIMv2,
    DeiT,
    DeiT3,
    MobileNetV4,
    PPOCRDetector,
    PPOCRRecognizer,
    RTDETRv1,
    YOLOv8,
    YOLOv10,
)
from trtutils.models._model import Model, _hw

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
    assert isinstance(model_cls._default_imgsz, (int, tuple))
    height, width = _hw(model_cls._default_imgsz)
    assert height > 0
    assert width > 0
    assert model_cls._dynamic_batch is False
    assert len(model_cls._input_tensors) > 0

    shapes = model_cls._make_shapes(4, model_cls._default_imgsz)
    assert len(shapes) == len(model_cls._input_tensors)
    for (name, kind), (shape_name, shape) in zip(model_cls._input_tensors, shapes):
        assert shape_name == name
        assert kind in ("image", "size")
        if kind == "image":
            assert shape == (4, 3, height, width)
        else:
            assert shape == (4, 2)


class _FakeModel(Model):
    _model_type = "fake"
    _friendly_name = "Fake"
    _default_imgsz = (256, 128)
    _input_tensors = [("input", "image"), ("sizes", "size")]  # noqa: RUF012
    _imgsz_divisor = 32


class _FakeDynamicModel(_FakeModel):
    _dynamic_batch = True


@pytest.mark.parametrize(
    ("imgsz", "expected"),
    [
        pytest.param(640, (2, 3, 640, 640), id="int"),
        pytest.param((256, 128), (2, 3, 256, 128), id="height-width"),
    ],
)
def test_make_shapes_height_width_order(imgsz, expected) -> None:
    """_make_shapes builds (batch, 3, height, width) from an int or an (H, W) tuple."""
    assert _FakeModel._make_shapes(2, imgsz) == [("input", expected), ("sizes", (2, 2))]


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
        pytest.param(RFDETR, (576, 288), id="divisor-hw"),
        pytest.param(YOLOv10, 123, id="unrestricted"),
        pytest.param(YOLOv10, (480, 640), id="unrestricted-hw"),
    ],
)
def test_validate_imgsz_accepts(model_cls, imgsz) -> None:
    """_validate_imgsz accepts sizes allowed by _valid_imgszs and _imgsz_divisor."""
    model_cls._validate_imgsz(imgsz)


@pytest.mark.parametrize(
    ("model_cls", "imgsz", "match"),
    [
        pytest.param(RTDETRv1, 320, "supports only imgsz", id="valid-list"),
        pytest.param(RTDETRv1, (640, 480), "supports only imgsz", id="valid-list-hw"),
        pytest.param(RFDETR, 577, "divisible by", id="divisor"),
        pytest.param(RFDETR, (576, 289), "divisible by", id="divisor-width"),
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
        pytest.param(YOLOv10, "yolov10n", (640, 640), 640, id="square-tuple"),
        pytest.param(YOLOv10, "yolov10n", (480, 640), (480, 640), id="height-width"),
        pytest.param(DeiT, "deit_tiny", (224, 112), (224, 112), id="deit-hw-patch"),
        pytest.param(MobileNetV4, "mobilenetv4_conv_small", None, 224, id="mnv4-small-default"),
        pytest.param(MobileNetV4, "mobilenetv4_hybrid_large", None, 384, id="mnv4-large-default"),
        pytest.param(MobileNetV4, "mobilenetv4_conv_medium", 320, 320, id="mnv4-test-size"),
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
        pytest.param(DeiT, "deit_tiny", (224, 120), "divisible by 16", id="deit-hw-patch"),
        pytest.param(
            MobileNetV4, "mobilenetv4_conv_small", 240, "divisible by 32", id="mnv4-stride"
        ),
        pytest.param(YOLOv10, "yolov10n", (640, 480, 3), "height, width", id="three-sizes"),
    ],
)
def test_download_rejects(tmp_path, model_cls, model, imgsz, match) -> None:
    """download() raises ValueError for unknown models and variant imgsz mismatches."""
    with pytest.raises(ValueError, match=match):
        model_cls.download(model, tmp_path / "out.onnx", imgsz=imgsz)


@pytest.mark.parametrize(
    ("config_imgsz", "imgsz", "expected"),
    [
        pytest.param(None, None, (256, 128), id="class-default"),
        pytest.param([384, 192], None, (384, 192), id="config-list"),
        pytest.param(["320", "320"], None, 320, id="config-square-strings"),
        pytest.param([384, 192], [512, 256], (512, 256), id="explicit-list"),
    ],
)
def test_download_height_width_defaults(tmp_path, config_imgsz, imgsz, expected) -> None:
    """download() takes (H, W) from the class default, a config list, or the caller."""
    config = {} if config_imgsz is None else {"imgsz": config_imgsz}
    with patch(
        "trtutils.models._model.load_model_configs", return_value={"fake": {"fake_m": config}}
    ), patch("trtutils.models._model.download_model_internal") as mock_download:
        _FakeModel.download("fake_m", tmp_path / "out.onnx", imgsz=imgsz)
    assert mock_download.call_args.kwargs["imgsz"] == expected


def test_build_unknown_kwargs_raises(tmp_path) -> None:
    """build() raises TypeError for keyword arguments no build hook accepts."""
    # yolov8's hook doesn't read the onnx, so the fake path is never opened
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        YOLOv8.build(tmp_path / "fake.onnx", tmp_path / "out.engine", totally_fake_kwarg=True)


@pytest.mark.parametrize(
    ("kwargs", "fp16"),
    [
        pytest.param({}, FLAGS.WEAK_TYPING, id="default"),
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
        pytest.param({}, FLAGS.WEAK_TYPING, id="default"),
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


@pytest.mark.parametrize("model_cls", [PPOCRDetector, PPOCRRecognizer])
def test_ppocr_build_rejects_height_width(model_cls: type[Model]) -> None:
    """PP-OCR profiles are built from one size, so a (height, width) imgsz raises before building."""
    with patch("trtutils.models.ocr._ppocr.build_engine") as build:
        with pytest.raises(TypeError, match="single imgsz"):
            model_cls.build("model.onnx", "model.engine", imgsz=(64, 32))
    build.assert_not_called()


@pytest.mark.parametrize("model_cls", [PPOCRDetector, PPOCRRecognizer])
@pytest.mark.parametrize("dynamic_batch", [True, False, None])
def test_ppocr_build_takes_dynamic_batch(model_cls: type[Model], dynamic_batch) -> None:
    """PP-OCR profiles are always dynamic: dynamic_batch is accepted and not sent to build_engine."""
    with patch("trtutils.models.ocr._ppocr.build_engine") as build:
        model_cls.build("model.onnx", "model.engine", dynamic_batch=dynamic_batch)
    assert "dynamic_batch" not in build.call_args.kwargs


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


def _built_shapes(model_cls, onnx_path, tmp_path, **kwargs: object):
    with patch("trtutils.models._model.build_engine") as mock_build:
        model_cls.build(onnx_path, tmp_path / "out.engine", **kwargs)
    return dict(mock_build.call_args.kwargs["shapes"])


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        pytest.param([1, 3, 384, 384], (384, 384), id="static"),
        pytest.param([1, 3, "h", "w"], (576, 576), id="symbolic-hw"),
        pytest.param([1, 3, 384, 320], (384, 320), id="non-square"),
        pytest.param([1, 384, 384, 3], (576, 576), id="nhwc"),
    ],
)
def test_build_infers_imgsz_from_onnx(tmp_path, shape, expected) -> None:
    """build() without imgsz uses the ONNX's static H/W, else the class default."""
    path = tmp_path / "m.onnx"
    _write_onnx(path, shape)
    shapes = _built_shapes(RFDETR, path, tmp_path)
    assert shapes["input"] == (1, 3, *expected)


@pytest.mark.parametrize(
    ("shape", "imgsz"),
    [
        pytest.param([1, 3, 384, 384], 384, id="int"),
        pytest.param([1, 3, 384, 384], (384, 384), id="square-tuple"),
        pytest.param([1, 3, 384, 320], (384, 320), id="height-width"),
    ],
)
def test_build_explicit_imgsz_matching_onnx(tmp_path, shape, imgsz) -> None:
    """An explicit imgsz equal to the ONNX size builds normally."""
    path = tmp_path / "m.onnx"
    _write_onnx(path, shape)
    assert _built_shapes(RFDETR, path, tmp_path, imgsz=imgsz)["input"] == tuple(shape)


@pytest.mark.parametrize(
    ("shape", "imgsz", "match"),
    [
        pytest.param([1, 3, 384, 384], 576, r"384.*576", id="square"),
        pytest.param([1, 3, 384, 320], (320, 384), r"\(384, 320\).*\(320, 384\)", id="swapped"),
    ],
)
def test_build_explicit_imgsz_contradicting_onnx_raises(tmp_path, shape, imgsz, match) -> None:
    """An explicit imgsz that differs from the static ONNX size raises before building."""
    path = tmp_path / "m.onnx"
    _write_onnx(path, shape)
    with patch("trtutils.models._model.build_engine") as mock_build:
        with pytest.raises(ValueError, match=match):
            RFDETR.build(path, tmp_path / "out.engine", imgsz=imgsz)
    mock_build.assert_not_called()


_STATIC = ((4, 3, 256, 128), (4, 2))
_DYNAMIC = (((1, 3, 256, 128), (4, 3, 256, 128), (4, 3, 256, 128)), ((1, 2), (4, 2), (4, 2)))


@pytest.mark.parametrize(
    ("model_cls", "kwargs", "expected"),
    [
        pytest.param(_FakeModel, {}, _STATIC, id="static"),
        pytest.param(_FakeModel, {"dynamic_batch": True}, _DYNAMIC, id="dynamic-override"),
        pytest.param(_FakeDynamicModel, {}, _DYNAMIC, id="dynamic-class"),
        pytest.param(_FakeDynamicModel, {"dynamic_batch": False}, _STATIC, id="static-override"),
        pytest.param(
            _FakeDynamicModel,
            {"embed_sst": True},
            (((1, 256, 128, 3), (4, 256, 128, 3), (4, 256, 128, 3)), _DYNAMIC[1]),
            id="dynamic-sst",
        ),
    ],
)
def test_build_profiles(tmp_path, model_cls, kwargs, expected) -> None:
    """build() takes the (H, W) default and gives every input a 1..batch_size profile if dynamic."""
    path = tmp_path / "m.onnx"
    _write_onnx(path, ["N", 3, "h", "w"])
    shapes = _built_shapes(model_cls, path, tmp_path, batch_size=4, **kwargs)
    assert (shapes["input"], shapes["sizes"]) == expected


@pytest.mark.parametrize(
    ("imgsz", "expected"),
    [
        pytest.param((256, 256), 256, id="square-tuple"),
        pytest.param((256, 128), (256, 128), id="height-width"),
    ],
)
def test_build_hooks_get_int_for_square(tmp_path, imgsz, expected) -> None:
    """Build hooks receive an int for a square size and (H, W) only for a non-square one."""
    hook = MagicMock(return_value={})
    path = tmp_path / "m.onnx"
    _write_onnx(path, ["N", 3, "h", "w"])
    with patch.object(_FakeModel, "_build_hooks", [hook]):
        _built_shapes(_FakeModel, path, tmp_path, imgsz=imgsz)
    assert hook.call_args.kwargs["imgsz"] == expected


@pytest.mark.parametrize(
    ("imgsz", "match"),
    [
        pytest.param((256, 100), "divisible by 32", id="divisor"),
        pytest.param((256, 0), "positive", id="zero"),
        pytest.param((256, 256, 256), "height, width", id="three-sizes"),
    ],
)
def test_build_rejects_imgsz(tmp_path, imgsz, match) -> None:
    """build() rejects (H, W) sizes off the divisor, non-positive, or of the wrong length."""
    path = tmp_path / "m.onnx"
    _write_onnx(path, ["N", 3, "h", "w"])
    with patch("trtutils.models._model.build_engine") as mock_build:
        with pytest.raises(ValueError, match=match):
            _FakeModel.build(path, tmp_path / "out.engine", imgsz=imgsz)
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


@pytest.mark.parametrize(
    "simplify",
    [
        pytest.param(True, id="true"),
        pytest.param(False, id="false"),
        pytest.param(None, id="none"),
        pytest.param(["onnxslim", "polygraphy"], id="tools"),
    ],
)
def test_download_forwards_simplify(simplify, tmp_path) -> None:
    """Model.download hands every accepted simplify value to the exporter unchanged."""
    with patch("trtutils.models._utils._download") as mock_dl:
        YOLOv8.download("yolov8n", tmp_path / "m.onnx", simplify=simplify)
    mock_dl.assert_called_once()
    assert mock_dl.call_args.kwargs["simplify"] == simplify


def test_download_simplify_defaults_to_true(tmp_path) -> None:
    """Model.download keeps simplifying by default."""
    with patch("trtutils.models._utils._download") as mock_dl:
        YOLOv8.download("yolov8n", tmp_path / "m.onnx")
    assert mock_dl.call_args.kwargs["simplify"] is True
