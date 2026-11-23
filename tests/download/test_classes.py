# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/download -- the classes option of open-vocabulary (YOLOE) exports."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tests.download.conftest import ROUTING_CASES, patch_exporter
from trtutils.download import download, download_model, load_model_configs
from trtutils.download._models import EXPORTERS, _ultralytics
from trtutils.download._tools import check_classes

pytestmark = pytest.mark.cpu

_NAMES = ["person", "traffic light", "horse"]
_FIXED = [p for p in ROUTING_CASES if not EXPORTERS[p.values[1]].classes]


@pytest.mark.parametrize(
    ("classes", "error", "match"),
    [
        pytest.param("person", TypeError, "one string", id="one-string"),
        pytest.param([], ValueError, "at least one", id="empty"),
        pytest.param(["person", " "], ValueError, "non-blank", id="blank"),
        pytest.param(["person", "car\n"], ValueError, "one line", id="newline"),
        pytest.param(["a\rb"], ValueError, "one line", id="carriage-return"),
        pytest.param(["person", "person"], ValueError, "unique", id="duplicate"),
    ],
)
def test_check_classes_rejects(classes, error, match) -> None:
    """Names must be a non-empty sequence of unique, non-blank, one-line strings."""
    with pytest.raises(error, match=match):
        check_classes(classes)


def test_check_classes_keeps_order() -> None:
    """Valid names come back as a list in the given order (the class id order)."""
    assert check_classes(tuple(_NAMES)) == _NAMES


@pytest.fixture
def route(fake_venv):
    """Run download_model with make_venv and one exporter mocked; yields (call, mock_venv)."""
    fake_python, fake_bin, fake_output = fake_venv
    with patch(
        "trtutils.download._download.make_venv", return_value=(fake_python, fake_bin)
    ) as mock_venv:

        def _call(model, exporter, tmp_path, **kwargs: object):
            with patch_exporter(exporter, fake_output) as mock_export:
                download_model(model, tmp_path, no_warn=True, **kwargs)
            return mock_export

        yield _call, mock_venv


@pytest.mark.parametrize(
    ("classes", "expected"),
    [
        pytest.param(None, None, id="default"),
        pytest.param(tuple(_NAMES), _NAMES, id="names"),
    ],
)
def test_text_prompt_exporter_gets_classes(route, tmp_path, classes, expected) -> None:
    """The YOLOE text-prompt exporter receives the checked names, or None for its default."""
    call, _ = route
    mock_export = call("yoloe-11s-seg", "yoloe", tmp_path, classes=classes)
    assert mock_export.call_args.kwargs.get("classes") == expected


@pytest.mark.parametrize(
    ("model", "exporter"),
    [*_FIXED, pytest.param("yoloe-11s-seg-pf", "yoloe_pf", id="yoloe-prompt-free")],
)
def test_fixed_vocabulary_rejects_classes(route, tmp_path, model, exporter) -> None:
    """Models without the classes capability refuse names before any work, and never see them."""
    call, mock_venv = route
    with pytest.raises(ValueError, match="fixed vocabulary"):
        call(model, exporter, tmp_path, classes=_NAMES)
    mock_venv.assert_not_called()
    assert "classes" not in call(model, exporter, tmp_path).call_args.kwargs


def test_invalid_classes_rejected_before_any_work(route, tmp_path) -> None:
    """Bad names fail before the venv is built."""
    call, mock_venv = route
    with pytest.raises(ValueError, match="unique"):
        call("yoloe-11s-seg", "yoloe", tmp_path, classes=["car", "car"])
    mock_venv.assert_not_called()


@pytest.mark.parametrize("names_file", [True, False], ids=["names", "no-names"])
def test_download_copies_class_names(tmp_path, names_file) -> None:
    """download() copies the exporter's names file next to the output as <stem>.classes.txt."""
    model = tmp_path / "work" / "yoloe-11s-seg.onnx"
    model.parent.mkdir()
    model.write_text("x")
    if names_file:
        model.with_suffix(".classes.txt").write_text("person\ncar\n")
    output = tmp_path / "out" / "yoloe_640.onnx"
    output.parent.mkdir()
    with patch("trtutils.download._download.check_uv_version"), patch(
        "trtutils.download._download.download_model", return_value=model
    ) as mock_dm:
        download("yoloe-11s-seg", output, classes=["person", "car"])
    assert mock_dm.call_args.kwargs["classes"] == ["person", "car"]
    assert output.exists()
    names = tmp_path / "out" / "yoloe_640.classes.txt"
    assert names.exists() is names_file
    if names_file:
        assert names.read_text().splitlines() == ["person", "car"]


@pytest.fixture
def export_calls(monkeypatch, tmp_path):
    """Stub the YOLOE exporter's tooling; run_cmd writes the ONNX and names the script would."""
    calls: list = []
    cache = tmp_path / "cache"
    cache.mkdir()

    def _run_cmd(argv: list[str], *, cwd, **_: object) -> None:
        calls.append(argv)
        (cwd / "upstream.onnx").write_text("onnx")
        (cwd / "upstream.classes.txt").write_text("a\nb\n")
        # ultralytics fetches the weights and the text encoder into the working directory
        for name in ("upstream.pt", "encoder.ts"):
            if not (cwd / name).exists():
                (cwd / name).write_text("fetched")

    monkeypatch.setattr(_ultralytics, "run_cmd", _run_cmd)
    monkeypatch.setattr(_ultralytics, "run_uv_pip_install", lambda *_, **__: None)
    monkeypatch.setattr(_ultralytics, "get_weights_cache_dir", lambda: cache)
    return calls, cache


def _export(func, model, tmp_path, **kwargs: object):
    config = next(c[model] for c in load_model_configs().values() if model in c)
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    config = {**config, "name": "upstream", "text_encoder": "encoder.ts"}
    path = func(work, config, work / "python", work, model, 17, no_warn=True, **kwargs)
    return path, (work / "_export_yoloe.py").read_text()


@pytest.mark.parametrize(
    ("model", "classes", "names", "end2end"),
    [
        pytest.param("yoloe-11s-seg", _NAMES, _NAMES, False, id="names"),
        pytest.param(
            "yoloe-11s-seg", None, list(_ultralytics.COCO80_NAMES), False, id="default-coco"
        ),
        pytest.param("yoloe-26n-seg", _NAMES, _NAMES, True, id="yoloe26-raw-head"),
    ],
)
def test_text_prompt_export_script(export_calls, tmp_path, model, classes, names, end2end) -> None:
    """The script embeds the names as text prompts, and the names file lands next to the ONNX."""
    calls, cache = export_calls
    path, script = _export(_ultralytics.export_yoloe, model, tmp_path, classes=classes)
    assert f"names = {names!r}\n" in script
    assert "model.set_classes(names, model.get_text_pe(names))" in script
    assert ("end2end=False" in script) is end2end
    assert (path.name, path.exists()) == (f"{model}.onnx", True)
    assert path.with_suffix(".classes.txt").read_text() == "a\nb\n"
    assert len(calls) == 1
    # the weights and the text encoder are cached once fetched
    assert {p.name for p in cache.iterdir()} == {"ultralytics_upstream.pt", "ultralytics_encoder.ts"}


def test_prompt_free_export_script(export_calls, tmp_path) -> None:
    """Prompt-free models keep their own vocabulary and never fetch the text encoder."""
    _calls, cache = export_calls
    path, script = _export(_ultralytics.export_yoloe_pf, "yoloe-11s-seg-pf", tmp_path)
    assert "names = None\n" in script
    assert path.with_suffix(".classes.txt").exists()
    assert {p.name for p in cache.iterdir()} == {"ultralytics_upstream.pt"}


def test_export_uses_cached_files(export_calls, tmp_path) -> None:
    """Cached weights and text encoder are copied in instead of fetched again."""
    _calls, cache = export_calls
    for name in ("upstream.pt", "encoder.ts"):
        (cache / f"ultralytics_{name}").write_text("cached")
    _export(_ultralytics.export_yoloe, "yoloe-11s-seg", tmp_path)
    work = tmp_path / "work"
    assert (work / "upstream.pt").read_text() == "cached"
    assert (work / "encoder.ts").read_text() == "cached"
