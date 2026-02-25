# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Parametrized model instantiation tests for all model classes."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_mock_engine(
    input_names: list = None,
    output_names: list = None,
    input_spec: list = None,
    output_spec: list = None,
) -> MagicMock:
    """Create a mock TRTEngine with configurable tensor names/specs."""
    engine = MagicMock()
    engine.input_names = input_names or ["images"]
    engine.output_names = output_names or [
        "num_dets",
        "det_boxes",
        "det_scores",
        "det_classes",
    ]
    engine.input_spec = input_spec or [("images", (1, 3, 640, 640))]
    engine.output_spec = output_spec or [
        ("num_dets", (1, 1)),
        ("det_boxes", (1, 100, 4)),
        ("det_scores", (1, 100)),
        ("det_classes", (1, 100)),
    ]
    return engine


# ---------------------------------------------------------------------------
# Model mixin class-attribute tests (no GPU needed)
# ---------------------------------------------------------------------------
DETECTOR_CLASSES = [
    "YOLOv3",
    "YOLOv5",
    "YOLOv7",
    "YOLOv8",
    "YOLOv9",
    "YOLOv10",
    "YOLOv11",
    "YOLOv12",
    "YOLOv13",
    "YOLOv26",
    "YOLOX",
    "RTDETRv1",
    "RTDETRv2",
    "RTDETRv3",
    "DFINE",
    "DEIM",
    "DEIMv2",
    "RFDETR",
]

CLASSIFIER_CLASSES = [
    "AlexNet",
    "ConvNeXt",
    "DenseNet",
    "EfficientNet",
    "EfficientNetV2",
    "GoogLeNet",
    "Inception",
    "MaxViT",
    "MNASNet",
    "MobileNetV2",
    "MobileNetV3",
    "RegNet",
    "ResNet",
    "ResNeXt",
    "ShuffleNetV2",
    "SqueezeNet",
    "SwinTransformer",
    "SwinTransformerV2",
    "VGG",
    "ViT",
    "WideResNet",
]

DEPTH_CLASSES = ["DepthAnythingV2"]

ALL_MODEL_NAMES = DETECTOR_CLASSES + CLASSIFIER_CLASSES + DEPTH_CLASSES


def _get_model_class(name: str) -> type:
    """Import and return a model class by name from trtutils.models."""
    import trtutils.models as models_mod

    return getattr(models_mod, name)


class TestModelClassAttributes:
    """Verify required class attributes are present on all Model subclasses."""

    @pytest.mark.cpu
    @pytest.mark.parametrize("cls_name", ALL_MODEL_NAMES)
    def test_has_model_type(self, cls_name: str) -> None:
        cls = _get_model_class(cls_name)
        assert hasattr(cls, "_model_type")
        assert isinstance(cls._model_type, str)
        assert len(cls._model_type) > 0

    @pytest.mark.cpu
    @pytest.mark.parametrize("cls_name", ALL_MODEL_NAMES)
    def test_has_friendly_name(self, cls_name: str) -> None:
        cls = _get_model_class(cls_name)
        assert hasattr(cls, "_friendly_name")
        assert isinstance(cls._friendly_name, str)

    @pytest.mark.cpu
    @pytest.mark.parametrize("cls_name", ALL_MODEL_NAMES)
    def test_has_default_imgsz(self, cls_name: str) -> None:
        cls = _get_model_class(cls_name)
        assert hasattr(cls, "_default_imgsz")
        assert isinstance(cls._default_imgsz, int)
        assert cls._default_imgsz > 0

    @pytest.mark.cpu
    @pytest.mark.parametrize("cls_name", ALL_MODEL_NAMES)
    def test_has_input_tensors(self, cls_name: str) -> None:
        cls = _get_model_class(cls_name)
        assert hasattr(cls, "_input_tensors")
        assert isinstance(cls._input_tensors, list)
        assert len(cls._input_tensors) > 0
        for name, kind in cls._input_tensors:
            assert isinstance(name, str)
            assert kind in ("image", "size")


class TestModelMakeShapes:
    """Test the _make_shapes classmethod."""

    @pytest.mark.cpu
    @pytest.mark.parametrize("cls_name", ALL_MODEL_NAMES)
    def test_make_shapes_returns_list(self, cls_name: str) -> None:
        cls = _get_model_class(cls_name)
        shapes = cls._make_shapes(1, cls._default_imgsz)
        assert isinstance(shapes, list)
        assert len(shapes) == len(cls._input_tensors)

    @pytest.mark.cpu
    @pytest.mark.parametrize("cls_name", ALL_MODEL_NAMES)
    def test_make_shapes_batch_size(self, cls_name: str) -> None:
        cls = _get_model_class(cls_name)
        batch = 4
        shapes = cls._make_shapes(batch, cls._default_imgsz)
        for name, shape in shapes:
            assert shape[0] == batch

    @pytest.mark.cpu
    @pytest.mark.parametrize("cls_name", ALL_MODEL_NAMES)
    def test_make_shapes_image_tensor_dims(self, cls_name: str) -> None:
        """Image tensors should have shape (B, 3, imgsz, imgsz)."""
        cls = _get_model_class(cls_name)
        imgsz = cls._default_imgsz
        shapes = cls._make_shapes(1, imgsz)
        for i, (name, kind) in enumerate(cls._input_tensors):
            _, shape = shapes[i]
            if kind == "image":
                assert len(shape) == 4
                assert shape[1] == 3
                assert shape[2] == imgsz
                assert shape[3] == imgsz
            elif kind == "size":
                assert len(shape) == 2
                assert shape[1] == 2


class TestModelValidateImgsz:
    """Test image size validation."""

    @pytest.mark.cpu
    def test_valid_imgszs_reject_invalid(self) -> None:
        """RTDETRv1 only allows 640; other sizes should raise."""
        cls = _get_model_class("RTDETRv1")
        with pytest.raises(ValueError, match="supports only imgsz"):
            cls._validate_imgsz(320)

    @pytest.mark.cpu
    def test_valid_imgszs_accept_valid(self) -> None:
        """RTDETRv1 should accept 640."""
        cls = _get_model_class("RTDETRv1")
        cls._validate_imgsz(640)  # Should not raise

    @pytest.mark.cpu
    def test_divisor_reject_invalid(self) -> None:
        """RFDETR requires imgsz divisible by 32."""
        cls = _get_model_class("RFDETR")
        with pytest.raises(ValueError, match="divisible by"):
            cls._validate_imgsz(577)

    @pytest.mark.cpu
    def test_divisor_accept_valid(self) -> None:
        """RFDETR should accept 576 (divisible by 32)."""
        cls = _get_model_class("RFDETR")
        cls._validate_imgsz(576)  # Should not raise

    @pytest.mark.cpu
    def test_no_restrictions_accept_any(self) -> None:
        """YOLOv10 has no imgsz restrictions."""
        cls = _get_model_class("YOLOv10")
        cls._validate_imgsz(123)  # Should not raise


class TestModelDownloadValidation:
    """Test download() class method validation (no actual downloads)."""

    @pytest.mark.cpu
    def test_invalid_model_name_raises(self, tmp_path: Path) -> None:
        """download() should raise for an invalid model name."""
        cls = _get_model_class("YOLOv10")
        out_path = tmp_path / "out.onnx"
        with pytest.raises(ValueError):
            cls.download("fake_model_xyz", out_path)

    @pytest.mark.cpu
    def test_deimv2_wrong_imgsz_raises(self, tmp_path: Path) -> None:
        """DEIMv2 with model-specific imgsz mismatch should raise."""
        cls = _get_model_class("DEIMv2")
        out_path = tmp_path / "out.onnx"
        with pytest.raises(ValueError):
            cls.download(
                "deimv2_atto",
                out_path,
                imgsz=640,
            )


class TestModelBuildValidation:
    """Test build() class method validation (no actual builds)."""

    @pytest.mark.cpu
    def test_unknown_kwargs_raises(self, tmp_path: Path) -> None:
        """build() should raise TypeError for unknown keyword args."""
        cls = _get_model_class("YOLOv10")
        fake_model_path = tmp_path / "fake.onnx"
        engine_path = tmp_path / "out.engine"
        with pytest.raises(TypeError, match="unexpected keyword arguments"):
            cls.build(
                fake_model_path,
                engine_path,
                totally_fake_kwarg=True,
            )


# ---------------------------------------------------------------------------
# Base arch classes (YOLO, DETR) -- no Model mixin
# ---------------------------------------------------------------------------
class TestBaseArchClasses:
    """Test that YOLO and DETR base classes exist and are importable."""

    @pytest.mark.cpu
    def test_yolo_importable(self) -> None:
        from trtutils.models import YOLO

        assert YOLO is not None

    @pytest.mark.cpu
    def test_detr_importable(self) -> None:
        from trtutils.models import DETR

        assert DETR is not None

    @pytest.mark.cpu
    def test_yolo_has_default_imgsz(self) -> None:
        from trtutils.models import YOLO

        assert hasattr(YOLO, "_default_imgsz")
        assert YOLO._default_imgsz == 640


# ---------------------------------------------------------------------------
# _make_shapes edge cases
# ---------------------------------------------------------------------------
class TestMakeShapesEdgeCases:
    """Test _make_shapes error handling."""

    @pytest.mark.cpu
    def test_unknown_tensor_kind_raises(self) -> None:
        """An unknown tensor kind should raise ValueError."""
        from trtutils.models._model import Model

        class FakeModel(Model):
            _model_type = "fake"
            _friendly_name = "Fake"
            _default_imgsz = 640
            _input_tensors = [("input", "unknown_kind")]

        with pytest.raises(ValueError, match="Unknown input tensor kind"):
            FakeModel._make_shapes(1, 640)


# ---------------------------------------------------------------------------
# Model.download() -- _model_imgszs variant-specific size logic
# ---------------------------------------------------------------------------
class TestModelImgszVariants:
    """Test the _model_imgszs logic in Model.download()."""

    @pytest.mark.cpu
    def test_deimv2_atto_default_imgsz_is_320(self, tmp_path: Path) -> None:
        """DEIMv2.download('deimv2_atto', ...) with imgsz=None should use 320."""
        from unittest.mock import patch as _patch

        cls = _get_model_class("DEIMv2")
        with _patch(
            "trtutils.models._model.download_model_internal",
        ) as mock_download:
            out_path = tmp_path / "out.onnx"
            cls.download("deimv2_atto", out_path)
            call_kwargs = mock_download.call_args[1]
            assert call_kwargs["imgsz"] == 320

    @pytest.mark.cpu
    def test_deimv2_femto_default_imgsz_is_416(self, tmp_path: Path) -> None:
        """DEIMv2.download('deimv2_femto', ...) with imgsz=None should use 416."""
        from unittest.mock import patch as _patch

        cls = _get_model_class("DEIMv2")
        with _patch(
            "trtutils.models._model.download_model_internal",
        ) as mock_download:
            out_path = tmp_path / "out.onnx"
            cls.download("deimv2_femto", out_path)
            call_kwargs = mock_download.call_args[1]
            assert call_kwargs["imgsz"] == 416

    @pytest.mark.cpu
    def test_deimv2_non_variant_default_imgsz_is_640(self, tmp_path: Path) -> None:
        """DEIMv2.download('deimv2_small', ...) with imgsz=None should use 640."""
        from unittest.mock import patch as _patch

        cls = _get_model_class("DEIMv2")
        with _patch(
            "trtutils.models._model.download_model_internal",
        ) as mock_download:
            out_path = tmp_path / "out.onnx"
            cls.download("deimv2_small", out_path)
            call_kwargs = mock_download.call_args[1]
            assert call_kwargs["imgsz"] == 640

    @pytest.mark.cpu
    def test_deimv2_atto_explicit_correct_imgsz_accepted(self, tmp_path: Path) -> None:
        """DEIMv2.download('deimv2_atto', ..., imgsz=320) should work."""
        from unittest.mock import patch as _patch

        cls = _get_model_class("DEIMv2")
        with _patch(
            "trtutils.models._model.download_model_internal",
        ) as mock_download:
            out_path = tmp_path / "out.onnx"
            cls.download("deimv2_atto", out_path, imgsz=320)
            call_kwargs = mock_download.call_args[1]
            assert call_kwargs["imgsz"] == 320

    @pytest.mark.cpu
    def test_deimv2_atto_wrong_imgsz_raises(self, tmp_path: Path) -> None:
        """DEIMv2.download('deimv2_atto', ..., imgsz=640) should raise."""
        cls = _get_model_class("DEIMv2")
        with pytest.raises(ValueError, match="requires imgsz of 320"):
            out_path = tmp_path / "out.onnx"
            cls.download("deimv2_atto", out_path, imgsz=640)


# ---------------------------------------------------------------------------
# Model.download() -- general imgsz=None path
# ---------------------------------------------------------------------------
class TestModelDownloadImgszDefault:
    """Test that download() uses _default_imgsz when imgsz=None."""

    @pytest.mark.cpu
    def test_yolov10_default_imgsz_640(self, tmp_path: Path) -> None:
        """YOLOv10.download with imgsz=None should use 640."""
        from unittest.mock import patch as _patch

        cls = _get_model_class("YOLOv10")
        with _patch(
            "trtutils.models._model.download_model_internal",
        ) as mock_download:
            out_path = tmp_path / "out.onnx"
            cls.download("yolov10n", out_path)
            call_kwargs = mock_download.call_args[1]
            assert call_kwargs["imgsz"] == 640

    @pytest.mark.cpu
    def test_rfdetr_default_imgsz_576(self, tmp_path: Path) -> None:
        """RFDETR.download with imgsz=None should use 576."""
        from unittest.mock import patch as _patch

        cls = _get_model_class("RFDETR")
        with _patch(
            "trtutils.models._model.download_model_internal",
        ) as mock_download:
            out_path = tmp_path / "out.onnx"
            cls.download("rfdetr_n", out_path)
            call_kwargs = mock_download.call_args[1]
            assert call_kwargs["imgsz"] == 576


# ---------------------------------------------------------------------------
# nms_build_hook
# ---------------------------------------------------------------------------
class TestNmsBuildHook:
    """Test the nms_build_hook function."""

    @pytest.mark.cpu
    def test_returns_hooks_list(self) -> None:
        """nms_build_hook should return a dict with 'hooks' key."""
        from trtutils.models._model import nms_build_hook

        result = nms_build_hook()
        assert "hooks" in result
        assert isinstance(result["hooks"], list)
        assert len(result["hooks"]) == 1

    @pytest.mark.cpu
    def test_custom_params_accepted(self) -> None:
        """nms_build_hook should accept custom NMS parameters."""
        from trtutils.models._model import nms_build_hook

        result = nms_build_hook(
            num_classes=10,
            conf_threshold=0.5,
            iou_threshold=0.7,
            top_k=50,
        )
        assert "hooks" in result
