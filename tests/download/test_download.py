# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
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

from .common import TEST_MODELS

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def _temporary_dir() -> Iterator[Path]:
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path)


def download_with_args(model: str) -> None:
    """Download a model with the given arguments."""
    with _temporary_dir() as temp_dir:
        output = temp_dir / "model.onnx"
        download(model, output, accept=True)
        assert output.exists()


@pytest.mark.parametrize("model", TEST_MODELS)
def test_download(model: str) -> None:
    """Download a model and verify the ONNX file is created."""
    download_with_args(model)
