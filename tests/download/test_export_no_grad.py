# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Exports never need gradients: every export script must switch autograd off."""

from __future__ import annotations

from pathlib import Path

import pytest

from trtutils.download._models import (
    _depth_anything,
    _hands23,
    _hoi_detr,
    _timm,
    _torchvision,
)
from trtutils.download._tools import get_patches_dir

pytestmark = pytest.mark.cpu

_NO_GRAD = "torch.set_grad_enabled(False)"

# patched upstream scripts; ultralytics, yolov9 and rfdetr already export under no_grad
_PATCHES = [
    "dfine_export_onnx.patch",
    "rtdetrv1_export_onnx.patch",
    "rtdetrv2_export_onnx.patch",
    "yolov7_export.patch",
    "yolox_export_onnx.patch",
]


@pytest.mark.parametrize("patch_name", _PATCHES)
def test_patch_disables_grad(patch_name: str) -> None:
    """Each upstream export patch adds the no-grad switch."""
    lines = (get_patches_dir() / patch_name).read_text().splitlines()
    assert any(line.startswith("+") and _NO_GRAD in line for line in lines)


@pytest.mark.parametrize(
    ("module", "count"),
    [
        pytest.param(_timm, 1, id="timm"),
        pytest.param(_torchvision, 1, id="torchvision"),
        pytest.param(_depth_anything, 2, id="depth_anything_v1_v2"),
        pytest.param(_hoi_detr, 1, id="hoi_detr"),
        pytest.param(_hands23, 1, id="hands23"),
    ],
)
def test_generated_script_disables_grad(module, count: int) -> None:
    """Each in-repo export script template switches autograd off before building the model."""
    assert Path(module.__file__).read_text(encoding="utf-8").count(_NO_GRAD) == count
