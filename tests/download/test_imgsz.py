# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for non-square imgsz handling in src/trtutils/download/ -- normalize_imgsz and hw."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tests.download.conftest import patch_exporter
from trtutils.download import download_model
from trtutils.download._models import EXPORTERS
from trtutils.download._tools import normalize_imgsz

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize(
    ("imgsz", "expected"),
    [
        pytest.param(640, 640, id="int"),
        pytest.param("224", 224, id="string"),
        pytest.param([320], 320, id="one-item"),
        pytest.param((640, 640), 640, id="square-tuple"),
        pytest.param([256, 128], (256, 128), id="height-width"),
        pytest.param(["256", "128"], (256, 128), id="config-strings"),
    ],
)
def test_normalize_imgsz(imgsz, expected) -> None:
    """Square sizes become an int and non-square ones a (height, width) tuple."""
    assert normalize_imgsz(imgsz) == expected


@pytest.mark.parametrize(
    "imgsz",
    [
        pytest.param(0, id="zero"),
        pytest.param((256, -1), id="negative"),
        pytest.param([], id="empty"),
        pytest.param((640, 480, 3), id="three-sizes"),
    ],
)
def test_normalize_imgsz_rejects(imgsz) -> None:
    """Sizes that are not one or two positive ints raise."""
    with pytest.raises(ValueError, match="one positive size or"):
        normalize_imgsz(imgsz)


@pytest.mark.parametrize(
    ("hw", "imgsz", "expected"),
    [
        pytest.param(False, (640, 640), 640, id="square-tuple"),
        pytest.param(False, [480], 480, id="one-item"),
        pytest.param(True, (480, 640), (480, 640), id="hw-exporter"),
        pytest.param(True, 640, 640, id="hw-exporter-int"),
    ],
)
def test_download_model_passes_imgsz(tmp_path, fake_venv, hw, imgsz, expected) -> None:
    """download_model hands exporters an int when square and (H, W) only to hw exporters."""
    fake_python, fake_bin, fake_output = fake_venv
    flags = {"yolox": EXPORTERS["yolox"]._replace(hw=hw)}
    with patch.dict(EXPORTERS, flags), patch(
        "trtutils.download._download.make_venv", return_value=(fake_python, fake_bin)
    ), patch_exporter("yolox", fake_output) as mock_export:
        download_model("yoloxn", tmp_path, imgsz=imgsz, no_warn=True)
    assert mock_export.call_args.args[6] == expected


def test_download_model_rejects_non_square_before_any_work(tmp_path) -> None:
    """A square-only exporter refuses a non-square size before creating the venv."""
    assert EXPORTERS["yolox"].hw is False
    with patch("trtutils.download._download.make_venv") as mock_venv:
        with pytest.raises(ValueError, match="exports only square images"):
            download_model("yoloxn", tmp_path, imgsz=(480, 640))
    mock_venv.assert_not_called()
