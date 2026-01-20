# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from trtutils.download import download

from .common import TEST_MODELS


def download_with_args(model: str) -> None:
    """Download a model with the given arguments."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "model.onnx"
        download(model, output, accept=True)
        assert output.exists()


@pytest.mark.parametrize("model", TEST_MODELS)
def test_download(model: str) -> None:
    """Download a model and verify the ONNX file is created."""
    download_with_args(model)
