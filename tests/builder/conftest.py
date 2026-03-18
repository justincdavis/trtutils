# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Builder test fixtures -- ONNX files, temp dirs, test images."""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

try:
    from trtutils.builder._build import build_engine
except ImportError:
    build_engine = None

DATA_DIR = Path(__file__).parent.parent.parent / "data"
ONNX_PATH = DATA_DIR / "simple.onnx"


@pytest.fixture(scope="session")
def onnx_path() -> Path:
    """Path to the test ONNX model."""
    if not ONNX_PATH.exists():
        pytest.skip("Test ONNX model not found")
    return ONNX_PATH


@pytest.fixture(scope="session")
def _can_build_engine(onnx_path: Path) -> bool:
    """Check if TRT can build engines on this hardware (session-cached)."""
    if build_engine is None:
        return False
    try:
        with tempfile.NamedTemporaryFile(suffix=".engine", delete=True) as f:
            build_engine(onnx_path, f.name, optimization_level=1)
            return True
    except RuntimeError:
        return False
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _skip_if_cannot_build(request: pytest.FixtureRequest, _can_build_engine: bool) -> None:
    """Skip builder tests requiring GPU if TRT cannot build engines."""
    if not request.node.get_closest_marker("cpu") and not _can_build_engine:
        pytest.skip("TRT does not support this GPU's compute capability")


@pytest.fixture
def output_engine_path(tmp_path: Path) -> Path:
    """Temporary path for built engine output."""
    return tmp_path / "test_output.engine"


@pytest.fixture
def test_image_dir(tmp_path: Path) -> Path:
    """Create a temp directory with synthetic test images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    rng = np.random.default_rng(42)
    for i in range(8):
        img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"test_{i:03d}.jpg"), img)
    return img_dir


@pytest.fixture
def small_image_dir(tmp_path: Path) -> Path:
    """Create a temp directory with fewer images (for batch boundary testing)."""
    img_dir = tmp_path / "small_images"
    img_dir.mkdir()
    rng = np.random.default_rng(99)
    for i in range(3):
        img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"img_{i:03d}.jpg"), img)
    return img_dir


@pytest.fixture
def single_image_dir(tmp_path: Path) -> Path:
    """Create a temp directory with exactly one image."""
    img_dir = tmp_path / "single_image"
    img_dir.mkdir()
    rng = np.random.default_rng(7)
    img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(img_dir / "only.jpg"), img)
    return img_dir


@pytest.fixture
def empty_dir(tmp_path: Path) -> Path:
    """Empty directory (no images)."""
    d = tmp_path / "empty"
    d.mkdir()
    return d


@pytest.fixture
def invalid_onnx_file(tmp_path: Path) -> Path:
    """Create a file with .onnx extension but invalid binary content."""
    p = tmp_path / "invalid.onnx"
    # Binary junk that triggers protobuf parse failure
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
def non_onnx_file(tmp_path: Path) -> Path:
    """Create a file with wrong extension."""
    p = tmp_path / "model.txt"
    p.write_text("not an onnx file")
    return p


@pytest.fixture
def calibration_cache_path(tmp_path: Path) -> Path:
    """Temporary path for calibration cache."""
    return tmp_path / "calibration.cache"


@pytest.fixture
def timing_cache_path(tmp_path: Path) -> Path:
    """Temporary path for timing cache."""
    return tmp_path / "timing.cache"


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Temporary directory for engine caching."""
    d = tmp_path / "cache"
    d.mkdir()
    return d
