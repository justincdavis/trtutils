# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Download test fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def download_tmp_dir(tmp_path: Path) -> Path:
    """Temporary directory for download operations."""
    d = tmp_path / "downloads"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="session")
def model_configs() -> dict:
    """Load model configurations once per session."""
    from trtutils.download import load_model_configs

    return load_model_configs()


@pytest.fixture(scope="session")
def supported_models() -> list:
    """Get all supported model names once per session."""
    from trtutils.download import get_supported_models

    return get_supported_models()
