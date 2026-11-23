# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/download/_download.py -- download API and helpers."""

from __future__ import annotations

import hashlib
import io
import zipfile
from contextlib import ExitStack
from unittest.mock import patch

import pytest

from tests.download.conftest import ROUTING_CASES, patch_exporter
from trtutils.download import download, download_model, load_model_configs
from trtutils.download._models import (
    export_deimv2,
    export_dfine,
    export_ppocr,
    export_rtdetrv4,
    export_rtmo,
    export_yolox,
    export_yunet,
)
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
        pytest.param("rtdetrv4", 4, id="rtdetrv4"),
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


def _zip_bytes(member: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(member, b"onnx")
    return buf.getvalue()


@pytest.mark.parametrize("match", [True, False], ids=["match", "mismatch"])
def test_export_rtmo_checks_sha256(tmp_path, match) -> None:
    """The RTMO exporter unzips and patches an archive whose sha256 matches and rejects others."""
    data = _zip_bytes("end2end.onnx")
    config = {
        "url": "https://example.com/files/rtmo.zip",
        "sha256": hashlib.sha256(data if match else b"other").hexdigest(),
    }
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "trtutils.download._models._rtmo.run_download",
                side_effect=lambda directory, *_a, **_k: (directory / "rtmo.zip").write_bytes(data),
            )
        )
        install = stack.enter_context(patch("trtutils.download._models._rtmo.run_uv_pip_install"))
        run = stack.enter_context(patch("trtutils.download._models._rtmo.run_cmd"))
        args = (tmp_path, config, tmp_path / "py", tmp_path / "bin", "rtmo_x", 17)
        if match:
            assert export_rtmo(*args) == tmp_path / "rtmo_x.onnx"
            assert (tmp_path / "end2end.onnx").read_bytes() == b"onnx"
            assert run.call_count == 1
        else:
            with pytest.raises(RuntimeError, match="sha256"):
                export_rtmo(*args)
            assert install.call_count == 0


@pytest.mark.parametrize("match", [True, False], ids=["match", "mismatch"])
def test_export_yunet_checks_sha256(tmp_path, match) -> None:
    """The YuNet exporter patches a download whose sha256 matches, at a size divisible by 32."""
    data = b"onnx"
    config = {
        "url": "https://example.com/files/yunet.onnx",
        "sha256": hashlib.sha256(data if match else b"other").hexdigest(),
    }
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "trtutils.download._models._yunet.run_download",
                side_effect=lambda directory, *_a, **_k: (directory / "yunet.onnx").write_bytes(
                    data
                ),
            )
        )
        install = stack.enter_context(patch("trtutils.download._models._yunet.run_uv_pip_install"))
        run = stack.enter_context(patch("trtutils.download._models._yunet.run_cmd"))
        args = (tmp_path, config, tmp_path / "py", tmp_path / "bin", "yunet_x", 17, 330)
        if match:
            assert export_yunet(*args) == tmp_path / "yunet_x.onnx"
            program = run.call_args.args[0][2]
            assert "S = 320" in program
            assert 'onnx.load("yunet.onnx")' in program
        else:
            with pytest.raises(RuntimeError, match="sha256"):
                export_yunet(*args)
            assert install.call_count == 0


@pytest.mark.parametrize(
    ("export_func", "family", "model", "config_path"),
    [
        pytest.param(
            export_dfine,
            "dfine",
            "dfine_s",
            "configs/dfine/objects365/dfine_hgnetv2_s_obj2coco.yml",
            id="dfine",
        ),
        pytest.param(
            export_rtdetrv4,
            "rtdetrv4",
            "rtdetrv4_s",
            "configs/rtv4/rtv4_hgnetv2_s_coco.yml",
            id="rtdetrv4",
        ),
    ],
)
def test_export_dfine_style_command(tmp_path, export_func, family, model, config_path) -> None:
    """D-FINE-style exporters run the upstream script on the family's config and checkpoint."""
    config = load_model_configs()[family][model]
    cmd = _exported_cmd(export_func, "_dfine", config, tmp_path, model)
    assert cmd[cmd.index("-c") + 1] == config_path
    assert cmd[cmd.index("-r") + 1] == config["name"] + ".pth"
    assert cmd[cmd.index("--imgsz") + 1] == "640"


@pytest.mark.parametrize(
    ("data", "match"),
    [
        pytest.param(b"onnx", True, id="match"),
        pytest.param(b"tampered", False, id="mismatch"),
    ],
)
def test_export_ppocr_checks_sha256(tmp_path, data, match) -> None:
    """The PP-OCR exporter renames a download whose sha256 matches the config and rejects others."""
    config = {
        "url": "https://example.com/files/model.onnx",
        "sha256": hashlib.sha256(b"onnx").hexdigest(),
    }
    with patch(
        "trtutils.download._models._ppocr.run_download",
        side_effect=lambda directory, *_a, **_k: (directory / "model.onnx").write_bytes(data),
    ):
        if match:
            path = export_ppocr(tmp_path, config, tmp_path / "py", tmp_path / "bin", "ppocr_x", 17)
            assert path == tmp_path / "ppocr_x.onnx"
            assert path.read_bytes() == data
        else:
            with pytest.raises(RuntimeError, match="sha256"):
                export_ppocr(tmp_path, config, tmp_path / "py", tmp_path / "bin", "ppocr_x", 17)


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
