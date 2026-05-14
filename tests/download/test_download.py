# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/download/_download.py -- download API and helpers."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from trtutils.download import download, download_model
from trtutils.download._simplify import simplify
from trtutils.download._tools import check_uv_available, handle_imgsz
from trtutils.models._utils import get_valid_models

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize(
    "func",
    [
        pytest.param(lambda p, m: download_model(m, p), id="download_model"),
        pytest.param(lambda p, m: download(m, p / "out.onnx"), id="download"),
    ],
)
@pytest.mark.parametrize("model", ["totally_fake_model_xyz", ""])
def test_unsupported_model_raises(tmp_path, func, model) -> None:
    """download_model and download both surface an error for unknown or empty names."""
    with pytest.raises((ValueError, RuntimeError)):
        func(tmp_path, model)


_ROUTING_CASES = [
    pytest.param("yolov8n", "export_ultralytics", id="ultralytics-yolov8"),
    pytest.param("yolov11n", "export_ultralytics", id="ultralytics-yolov11"),
    pytest.param("yolov7t", "export_yolov7", id="yolov7"),
    pytest.param("yolov9t", "export_yolov9", id="yolov9"),
    pytest.param("yolov10n", "export_yolov10", id="yolov10"),
    pytest.param("yolov12n", "export_yolov12", id="yolov12"),
    pytest.param("yolov13n", "export_yolov13", id="yolov13"),
    pytest.param("yoloxn", "export_yolox", id="yolox"),
    pytest.param("rtdetrv1_r18", "export_rtdetrv1", id="rtdetrv1"),
    pytest.param("rtdetrv2_r18", "export_rtdetrv2", id="rtdetrv2"),
    pytest.param("rtdetrv3_r18", "export_rtdetrv3", id="rtdetrv3"),
    pytest.param("dfine_n", "export_dfine", id="dfine"),
    pytest.param("deim_dfine_n", "export_deim", id="deim"),
    pytest.param("deimv2_atto", "export_deimv2", id="deimv2"),
    pytest.param("rfdetr_n", "export_rfdetr", id="rfdetr"),
    pytest.param("alexnet", "export_torchvision_classifier", id="torchvision"),
    pytest.param("depth_anything_v2_small", "export_depth_anything_v2", id="depth_anything"),
]


@pytest.mark.parametrize(("model", "export_func_name"), _ROUTING_CASES)
def test_export_routing(tmp_path, fake_venv, model, export_func_name) -> None:
    """download_model routes each model to its corresponding export function."""
    fake_python, fake_bin, fake_output = fake_venv
    with patch(
        "trtutils.download._download.make_venv",
        return_value=(fake_python, fake_bin),
    ), patch(
        f"trtutils.download._download.{export_func_name}",
        return_value=fake_output,
    ) as mock_export:
        download_model(model, tmp_path)
        mock_export.assert_called_once()


@pytest.mark.parametrize(
    ("simplify_arg", "expected_tools"),
    [
        pytest.param(True, None, id="bool-true"),
        pytest.param(["onnxslim"], ["onnxslim"], id="list"),
        pytest.param("onnxslim", ["onnxslim"], id="string"),
    ],
)
@pytest.mark.usefixtures("patched_yolov10_export")
def test_simplify_arg_routing(tmp_path, simplify_arg, expected_tools) -> None:
    """download_model normalises the simplify argument before invoking _simplify.simplify."""
    with patch("trtutils.download._download._simplify.simplify") as mock_simplify:
        download_model("yolov10n", tmp_path, simplify=simplify_arg)
        mock_simplify.assert_called_once()
        assert mock_simplify.call_args.kwargs["tools"] == expected_tools


@pytest.mark.usefixtures("patched_yolov10_export")
def test_requirements_export_invokes_export_requirements(tmp_path) -> None:
    """Passing requirements_export triggers export_requirements after the export call."""
    req_path = tmp_path / "requirements.txt"
    with patch("trtutils.download._download.export_requirements") as mock_export_req:
        download_model("yolov10n", tmp_path, requirements_export=req_path)
        mock_export_req.assert_called_once()


def test_download_calls_download_model_and_copies(tmp_path) -> None:
    """download() invokes download_model in a temp dir and copies output to the requested path."""
    output_path = tmp_path / "output.onnx"
    with patch(
        "trtutils.download._download.check_uv_version",
    ), patch(
        "trtutils.download._download.download_model",
        return_value=tmp_path / "fake" / "model.onnx",
    ) as mock_dm, patch(
        "trtutils.download._download.shutil.copy",
    ) as mock_copy:
        download("yolov10n", output_path)
        mock_dm.assert_called_once()
        mock_copy.assert_called_once()


@pytest.mark.parametrize(
    ("imgsz", "default", "kwargs", "expected"),
    [
        pytest.param(None, 640, {}, 640, id="default"),
        pytest.param(320, 640, {}, 320, id="passthrough"),
        pytest.param(320, 640, {"enforce": True}, 640, id="enforce-overrides"),
        pytest.param(640, 640, {"enforce": True}, 640, id="enforce-noop"),
        pytest.param(641, 640, {"adjust_div": 32}, 640, id="adjust-rounds-down"),
        pytest.param(640, 640, {"adjust_div": 32}, 640, id="adjust-already-divisible"),
    ],
)
def test_handle_imgsz(imgsz, default, kwargs, expected) -> None:
    """handle_imgsz applies enforce/adjust_div policies to the requested image size."""
    assert handle_imgsz(imgsz, default, "test_model", **kwargs) == expected


def test_check_uv_available_no_uv_raises() -> None:
    """check_uv_available raises RuntimeError when uv is not on PATH."""
    with patch("shutil.which", return_value=None), pytest.raises(
        RuntimeError, match="uv is not available"
    ):
        check_uv_available()


def test_simplify_unknown_tool_raises(tmp_path) -> None:
    """simplify() rejects unknown tool names with ValueError."""
    with pytest.raises(ValueError, match="Unknown simplification tools"):
        simplify(tmp_path / "fake.onnx", tools=["not_a_real_tool"])


@pytest.mark.parametrize(
    ("model_type", "expected_count"),
    [
        pytest.param("yolov8", 5, id="yolov8"),
        pytest.param("yolov10", 11, id="yolov10"),
        pytest.param("rtdetrv1", 7, id="rtdetrv1"),
        pytest.param("rfdetr", 3, id="rfdetr"),
        pytest.param("deimv2", 8, id="deimv2"),
        pytest.param("yolox", 7, id="yolox"),
        pytest.param("dfine", 5, id="dfine"),
    ],
)
def test_get_valid_models_count_per_family(model_type, expected_count) -> None:
    """get_valid_models returns the expected count of models for each family."""
    assert len(get_valid_models(model_type)) == expected_count


def test_get_valid_models_invalid_family_raises() -> None:
    """get_valid_models raises KeyError for an unknown family."""
    with pytest.raises(KeyError):
        get_valid_models("nonexistent_family")
