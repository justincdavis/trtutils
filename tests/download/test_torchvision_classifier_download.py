# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="misc"
from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from trtutils.download import download
from trtutils.models._utils import get_valid_models

if TYPE_CHECKING:
    from collections.abc import Iterator

ALL_TORCHVISION_MODELS = get_valid_models("torchvision_classifier")

# Representative subset: one from each model family for fast CI
REPRESENTATIVE_MODELS = [
    "alexnet",
    "convnext_tiny",
    "densenet121",
    "efficientnet_b0",
    "efficientnet_v2_s",
    "googlenet",
    "inception_v3",
    "maxvit_t",
    "mnasnet0_5",
    "mobilenet_v2",
    "mobilenet_v3_small",
    "regnet_y_400mf",
    "resnet18",
    "resnext50_32x4d",
    "shufflenet_v2_x0_5",
    "squeezenet1_0",
    "swin_t",
    "swin_v2_t",
    "vgg11",
    "vit_b_16",
    "wide_resnet50_2",
]


@contextmanager
def _temporary_dir() -> Iterator[Path]:
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path)


@pytest.mark.parametrize("model", REPRESENTATIVE_MODELS)
def test_download_torchvision_classifier(model: str) -> None:
    """Download a representative torchvision classifier model and verify the ONNX file is created."""
    with _temporary_dir() as temp_dir:
        output = temp_dir / f"{model}.onnx"
        download(model, output, accept=True)
        assert output.exists(), f"ONNX file was not created for {model}"
        assert output.stat().st_size > 0, f"ONNX file is empty for {model}"


@pytest.mark.parametrize("model", ALL_TORCHVISION_MODELS)
def test_download_all_torchvision_classifiers(model: str) -> None:
    """Download every torchvision classifier model and verify the ONNX file is created."""
    with _temporary_dir() as temp_dir:
        output = temp_dir / f"{model}.onnx"
        download(model, output, accept=True)
        assert output.exists(), f"ONNX file was not created for {model}"
        assert output.stat().st_size > 0, f"ONNX file is empty for {model}"
