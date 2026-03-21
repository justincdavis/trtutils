# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Builder test fixtures -- ONNX files, temp dirs, test images."""

from __future__ import annotations

import os
import struct
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

_CPU_ONLY = os.environ.get("TRTUTILS_IGNORE_MISSING_CUDA", "0") == "1"


def _make_image_dir(base: Path, name: str, count: int, seed: int) -> Path:
    """Create a temp directory with synthetic test images."""
    img_dir = base / name
    img_dir.mkdir()
    rng = np.random.default_rng(seed)
    for i in range(count):
        img = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"img_{i:03d}.jpg"), img)
    return img_dir


@pytest.fixture
def output_engine_path(tmp_path) -> Path:
    """Temporary path for built engine output."""
    return tmp_path / "test_output.engine"


@pytest.fixture(scope="session")
def test_image_dir(tmp_path_factory) -> Path:
    """Directory with 8 synthetic test images."""
    return _make_image_dir(tmp_path_factory.mktemp("images"), "imgs", 8, 42)


@pytest.fixture(scope="session")
def single_image_dir(tmp_path_factory) -> Path:
    """Directory with exactly one image."""
    return _make_image_dir(tmp_path_factory.mktemp("single"), "imgs", 1, 7)


@pytest.fixture
def empty_dir(tmp_path) -> Path:
    """Empty directory (no images)."""
    d = tmp_path / "empty"
    d.mkdir()
    return d


@pytest.fixture
def invalid_onnx_file(tmp_path) -> Path:
    """Create a file with .onnx extension but invalid binary content."""
    p = tmp_path / "invalid.onnx"
    p.write_bytes(
        struct.pack(
            ">IIIIII",
            0xDEADBEEF,
            0xCAFEBABE,
            0xBAADF00D,
            0x12345678,
            0xFEEDFACE,
            0xDECAF000,
        )
    )
    return p


@pytest.fixture
def non_onnx_file(tmp_path) -> Path:
    """Create a file with wrong extension."""
    p = tmp_path / "model.txt"
    p.write_text("not an onnx file")
    return p


@pytest.fixture
def calibration_cache_path(tmp_path) -> Path:
    """Temporary path for calibration cache."""
    return tmp_path / "calibration.cache"


@pytest.fixture
def timing_cache_path(tmp_path) -> Path:
    """Temporary path for timing cache."""
    return tmp_path / "timing.cache"


@pytest.fixture
def cache_dir(tmp_path) -> Path:
    """Temporary directory for engine caching."""
    d = tmp_path / "cache"
    d.mkdir()
    return d


if not _CPU_ONLY:
    from tests.builder._gpu_fixtures import *  # noqa: F403
