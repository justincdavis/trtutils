# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for the batch / dynamic export options of download_model and the exporters."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from tests.download.conftest import ROUTING_CASES
from trtutils.download import download, download_model, load_model_configs
from trtutils.download._models import _torchvision, _ultralytics, _yolo

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.cpu

_BATCH_EXPORTERS = {
    "export_ultralytics",
    "export_yolov10",
    "export_yolov12",
    "export_yolov13",
    "export_torchvision_classifier",
}
_SUPPORTED = [p for p in ROUTING_CASES if p.values[1] in _BATCH_EXPORTERS]
_UNSUPPORTED = [p for p in ROUTING_CASES if p.values[1] not in _BATCH_EXPORTERS]


@pytest.fixture
def route(fake_venv):
    """Run download_model with make_venv and one exporter mocked; yields (call, mock_venv)."""
    fake_python, fake_bin, fake_output = fake_venv
    with patch(
        "trtutils.download._download.make_venv", return_value=(fake_python, fake_bin)
    ) as mock_venv:

        def _call(model, export_func_name, tmp_path, **kwargs: object):
            with patch(
                f"trtutils.download._download.{export_func_name}", return_value=fake_output
            ) as mock_export:
                download_model(model, tmp_path, **kwargs)
            return mock_export

        yield _call, mock_venv


@pytest.mark.parametrize(
    ("kwargs", "expected", "expected_torchvision"),
    [
        pytest.param({}, (1, False), (1, True), id="default"),
        pytest.param({"batch": 1}, (1, False), (1, False), id="batch-1"),
        pytest.param({"batch": 8}, (8, False), (8, False), id="batch-8"),
        pytest.param({"dynamic": True}, (1, True), (1, True), id="dynamic"),
        pytest.param({"dynamic": False}, (1, False), (1, False), id="static"),
        pytest.param({"batch": 1, "dynamic": True}, (1, True), (1, True), id="dynamic-b1"),
    ],
)
@pytest.mark.parametrize(("model", "export_func_name"), _SUPPORTED)
def test_batch_exporters_get_resolved_options(
    route, tmp_path, model, export_func_name, kwargs, expected, expected_torchvision
) -> None:
    """Batch-capable exporters receive a concrete batch and dynamic; torchvision defaults dynamic."""
    call, _ = route
    mock_export = call(model, export_func_name, tmp_path, **kwargs)
    if export_func_name == "export_torchvision_classifier":
        expected = expected_torchvision
    got = mock_export.call_args.kwargs
    assert (got["batch"], got["dynamic"]) == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="default"),
        pytest.param({"batch": 1}, id="batch-1"),
        pytest.param({"dynamic": False}, id="static"),
    ],
)
@pytest.mark.parametrize(("model", "export_func_name"), _UNSUPPORTED)
def test_other_exporters_never_see_batch_options(
    route, tmp_path, model, export_func_name, kwargs
) -> None:
    """Exporters that cannot batch export as before, without batch or dynamic arguments."""
    call, _ = route
    mock_export = call(model, export_func_name, tmp_path, **kwargs)
    assert {"batch", "dynamic"}.isdisjoint(mock_export.call_args.kwargs)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        pytest.param({"batch": 2}, NotImplementedError, id="batch-2"),
        pytest.param({"dynamic": True}, NotImplementedError, id="dynamic"),
        pytest.param({"batch": 4, "dynamic": True}, NotImplementedError, id="batch-and-dynamic"),
        pytest.param({"batch": 0}, ValueError, id="zero-batch"),
    ],
)
@pytest.mark.parametrize(("model", "export_func_name"), _UNSUPPORTED)
def test_other_exporters_refuse_before_any_work(
    route, tmp_path, model, export_func_name, kwargs, error
) -> None:
    """Families that cannot batch refuse up front instead of silently exporting at batch 1."""
    call, mock_venv = route
    with pytest.raises(error):
        call(model, export_func_name, tmp_path, **kwargs)
    mock_venv.assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"batch": 4, "dynamic": True}, id="batch-and-dynamic"),
        pytest.param({"batch": 0}, id="zero-batch"),
        pytest.param({"batch": -2}, id="negative-batch"),
        pytest.param({"dynamic": True, "make_static": True}, id="dynamic-and-make-static"),
    ],
)
@pytest.mark.parametrize(("model", "export_func_name"), _SUPPORTED)
def test_invalid_options_rejected_before_any_work(
    route, tmp_path, model, export_func_name, kwargs
) -> None:
    """Contradictory or non-positive options raise ValueError before the venv is built."""
    call, mock_venv = route
    with pytest.raises(ValueError, match=r"batch|make_static|positive"):
        call(model, export_func_name, tmp_path, **kwargs)
    mock_venv.assert_not_called()


@pytest.mark.parametrize(("batch", "dynamic"), [(None, None), (8, None), (None, True), (1, False)])
def test_download_forwards_batch_and_dynamic(tmp_path, batch, dynamic) -> None:
    """download() passes batch and dynamic through to download_model as keywords."""
    with patch("trtutils.download._download.check_uv_version"), patch(
        "trtutils.download._download.download_model", return_value=tmp_path / "m.onnx"
    ) as mock_dm, patch("trtutils.download._download.shutil.copy"):
        download("yolov10n", tmp_path / "out.onnx", batch=batch, dynamic=dynamic)
    assert mock_dm.call_args.kwargs["batch"] == batch
    assert mock_dm.call_args.kwargs["dynamic"] is dynamic


@pytest.fixture
def run_cmd_calls(monkeypatch, tmp_path):
    """Stub the batch exporters' tooling; run_cmd records its argv and writes the ONNX it names."""
    calls: list = []

    def _run_cmd(argv: list[str], *, cwd: Path, **_: object) -> None:
        calls.append(argv)
        cwd.mkdir(parents=True, exist_ok=True)
        (cwd / "out.onnx").touch()

    for module in (_torchvision, _ultralytics, _yolo):
        monkeypatch.setattr(module, "run_cmd", _run_cmd)
        for name in ("git_clone", "run_uv_pip_install", "run_download"):
            monkeypatch.setattr(module, name, lambda *_, **__: None, raising=False)
    monkeypatch.setattr(_ultralytics, "get_weights_cache_dir", lambda: tmp_path)
    return calls


def _export(func, model, tmp_path, **kwargs: object):
    config = next(c[model] for c in load_model_configs().values() if model in c)
    bin_path = tmp_path / "bin"
    return func(
        tmp_path,
        {**config, "name": "out"},
        bin_path / "python",
        bin_path,
        model,
        17,
        no_warn=True,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("batch", "dynamic"), [(1, False), (8, False), (1, True)], ids=["b1", "b8", "dynamic"]
)
@pytest.mark.parametrize(
    ("func", "model"),
    [
        pytest.param(_ultralytics.export_ultralytics, "yolov8n", id="ultralytics"),
        pytest.param(_yolo.export_yolov10, "yolov10n", id="yolov10"),
        pytest.param(_yolo.export_yolov12, "yolov12n", id="yolov12"),
        pytest.param(_yolo.export_yolov13, "yolov13n", id="yolov13"),
    ],
)
def test_yolo_cli_exporters_pass_batch_and_dynamic(
    run_cmd_calls, tmp_path, func, model, batch, dynamic
) -> None:
    """Exporters built on the ultralytics CLI hand batch= and dynamic= to `yolo export`."""
    _export(func, model, tmp_path, batch=batch, dynamic=dynamic)
    (argv,) = run_cmd_calls
    assert [a for a in argv if a.startswith(("batch=", "dynamic="))] == [
        f"batch={batch}",
        f"dynamic={dynamic}",
    ]


@pytest.mark.parametrize(
    ("batch", "dynamic"), [(1, False), (4, False), (1, True)], ids=["b1", "b4", "dynamic"]
)
@pytest.mark.usefixtures("run_cmd_calls")
def test_torchvision_export_script(tmp_path, batch, dynamic) -> None:
    """The torchvision classifier traces at the batch and leaves it symbolic only when dynamic."""
    _export(
        _torchvision.export_torchvision_classifier,
        "resnet18",
        tmp_path,
        batch=batch,
        dynamic=dynamic,
    )
    script = (tmp_path / "_export_torchvision_classifier.py").read_text()
    assert f"torch.randn({batch}, 3, imgsz, imgsz)" in script
    axes = '{"input": {0: "batch_size"}, "output": {0: "batch_size"}}' if dynamic else "None"
    assert f"dynamic_axes={axes}," in script
