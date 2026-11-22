# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/download/_download.py -- download API and helpers."""

from __future__ import annotations

import subprocess
import sys
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import onnx
import pytest
from onnx import helper, numpy_helper

from tests.download.conftest import ROUTING_CASES, patch_exporter
from trtutils.download import download, download_model, load_model_configs
from trtutils.download._make_static import _MAKE_STATIC_SCRIPT
from trtutils.download._models import export_deimv2, export_yolox
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


@pytest.mark.parametrize(("model", "exporter"), ROUTING_CASES)
def test_export_routing(tmp_path, fake_venv, model, exporter) -> None:
    """download_model routes each model to the EXPORTERS entry its config names."""
    fake_python, fake_bin, fake_output = fake_venv
    with patch(
        "trtutils.download._download.make_venv",
        return_value=(fake_python, fake_bin),
    ), patch_exporter(exporter, fake_output) as mock_export:
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


@pytest.mark.parametrize("external", [False, True], ids=["inline", "external-data"])
def test_simplify_skipped_for_external_data(tmp_path, fake_venv, external) -> None:
    """A model whose weights the exporter put in a .data sidecar (over 2GB) is not simplified."""
    fake_python, fake_bin, fake_output = fake_venv
    if external:
        fake_output.with_name("model.onnx.data").touch()
    with patch(
        "trtutils.download._download.make_venv", return_value=(fake_python, fake_bin)
    ), patch_exporter("timm_features", fake_output), patch(
        "trtutils.download._download._simplify.simplify"
    ) as mock_simplify:
        download_model("dinov3_vith16plus", tmp_path, simplify=True, no_warn=True)
    assert mock_simplify.called is not external


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


def test_download_accepts_str_output(tmp_path) -> None:
    """A str output (Python API) works: sidecar copy uses output.parent."""
    model = tmp_path / "fake" / "model.onnx"
    model.parent.mkdir()
    model.write_text("x")
    model.with_name("model.onnx.data").write_text("w")
    output = tmp_path / "out" / "output.onnx"
    output.parent.mkdir()
    with patch("trtutils.download._download.check_uv_version"), patch(
        "trtutils.download._download.download_model", return_value=model
    ):
        download("yolov10n", str(output))
    assert output.exists()
    assert (output.parent / "model.onnx.data").exists()


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


def _exported_cmd(export_func, module, config, tmp_path, model) -> list[str]:
    """Run an exporter with everything but the export command mocked, return that command."""
    with ExitStack() as stack:
        for name in ("git_clone", "run_patch", "run_uv_pip_install", "run_download", "shutil.move"):
            stack.enter_context(patch(f"trtutils.download._models.{module}.{name}"))
        run_cmd = stack.enter_context(patch(f"trtutils.download._models.{module}.run_cmd"))
        export_func(tmp_path, config, tmp_path / "py", tmp_path / "bin", model, 17)
    return [str(part) for part in run_cmd.call_args.args[0]]


@pytest.mark.parametrize(
    ("model", "exp_name"),
    [
        pytest.param("yoloxs", "yolox_s", id="yolox_s"),
        pytest.param("yolox_darknet", "yolov3", id="darknet-uses-yolov3-exp"),
    ],
)
def test_export_yolox_exp_file(tmp_path, model, exp_name) -> None:
    """The YOLOX exporter picks the upstream exp file, darknet's is yolov3.py."""
    config = load_model_configs()["yolox"][model]
    cmd = _exported_cmd(export_yolox, "_yolox", config, tmp_path, model)
    assert cmd[cmd.index("-f") + 1].endswith(f"exps/default/{exp_name}.py")


def test_export_deimv2_skips_onnxsim(tmp_path) -> None:
    """DEIMv2's own --simplify (onnxsim) crashes on the L/X rope backbone, trtutils simplifies after."""
    config = load_model_configs()["deimv2"]["deimv2_l"]
    cmd = _exported_cmd(export_deimv2, "_deim", config, tmp_path, "deimv2_l")
    assert "--simplify" not in cmd


@pytest.mark.parametrize(
    ("model", "exporter", "expected"),
    [
        pytest.param("yoloxn", "yolox", "Apache-2.0 licensed, be aware", id="permissive"),
        pytest.param("yolov8n", "ultralytics", "AGPL-3.0 licensed: a RESTRICTIVE", id="agpl"),
        pytest.param("yolov7t", "yolov7", "GPL-3.0 licensed: a RESTRICTIVE", id="gpl"),
        pytest.param(
            "depth_anything_v2_base",
            "depth_anything_v2",
            "CC-BY-NC-4.0 licensed: a RESTRICTIVE",
            id="non-commercial",
        ),
        pytest.param(
            "deimv2_atto", "deimv2", "(non-commercial) licensed: a RESTRICTIVE", id="custom"
        ),
        pytest.param("hands23_x101", "hands23", "not stated upstream", id="unknown"),
    ],
)
@pytest.mark.parametrize("no_warn", [False, True], ids=["warn", "no-warn"])
def test_license_notice(tmp_path, fake_venv, model, exporter, expected, no_warn) -> None:
    """download_model logs one license notice from the config, louder for restrictive ones."""
    fake_python, fake_bin, fake_output = fake_venv
    with patch(
        "trtutils.download._download.make_venv", return_value=(fake_python, fake_bin)
    ), patch_exporter(exporter, fake_output), patch(
        "trtutils.download._download.LOG.warning"
    ) as mock_warning:
        download_model(model, tmp_path, no_warn=no_warn)
    if no_warn:
        mock_warning.assert_not_called()
    else:
        (call,) = mock_warning.call_args_list
        assert model in call.args[0]
        assert expected in call.args[0]


def test_make_static_keeps_external_data(tmp_path) -> None:
    """make_static rewrites only the graph of a model whose weights live in a .data sidecar."""
    weight = numpy_helper.from_array(np.arange(4096, dtype=np.float32), name="weight")
    graph = helper.make_graph(
        [helper.make_node("Add", ["input", "weight"], ["output"])],
        "external",
        [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, ["batch", 4096])],
        [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, ["batch", 4096])],
        [weight],
    )
    model_path = tmp_path / "model.onnx"
    onnx.save(
        helper.make_model(graph),
        model_path,
        save_as_external_data=True,
        location="model.onnx.data",
        size_threshold=0,
    )
    sidecar = (tmp_path / "model.onnx.data").read_bytes()

    subprocess.run(  # noqa: S603
        [sys.executable, "-c", _MAKE_STATIC_SCRIPT, str(model_path)], check=True
    )

    assert (tmp_path / "model.onnx.data").read_bytes() == sidecar
    static = onnx.load(str(model_path))
    assert [d.dim_value for d in static.graph.input[0].type.tensor_type.shape.dim] == [1, 4096]
    np.testing.assert_array_equal(
        numpy_helper.to_array(static.graph.initializer[0]), np.arange(4096, dtype=np.float32)
    )
