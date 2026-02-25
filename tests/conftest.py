"""Root conftest.py — shared fixtures for the trtutils test suite."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR
ENGINES_DIR = DATA_DIR / "engines"

# Path objects (not str) — use str(path) when passing to cv2/etc.
HORSE_IMAGE_PATH = IMAGES_DIR / "horse.jpg"
PEOPLE_IMAGE_PATH = IMAGES_DIR / "people.jpeg"
IMAGE_PATHS = [HORSE_IMAGE_PATH, PEOPLE_IMAGE_PATH]


# ---------------------------------------------------------------------------
# TRT version detection (for engine path versioning)
# ---------------------------------------------------------------------------
def get_trt_version() -> str:
    """Get TensorRT version string for engine path caching."""
    try:
        from trtutils.compat._libs import trt

        return trt.__version__
    except Exception:
        return "unknown"


TRT_VERSION = get_trt_version()


def version_engine_path(path: Path) -> Path:
    """Insert TRT version into engine filename for caching."""
    return path.with_name(f"{path.stem}_{TRT_VERSION}{path.suffix}")


# ---------------------------------------------------------------------------
# GPU availability
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def gpu_available() -> bool:
    """Check if a CUDA GPU is available."""
    try:
        from trtutils.core import get_device_count

        return get_device_count() > 0
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _skip_gpu_tests(request: pytest.FixtureRequest, gpu_available: bool) -> None:
    """Auto-skip tests marked @pytest.mark.gpu when no GPU is available."""
    if request.node.get_closest_marker("gpu") and not gpu_available:
        pytest.skip("No CUDA GPU available")


@pytest.fixture(autouse=True)
def _skip_dla_tests(request: pytest.FixtureRequest) -> None:
    """Auto-skip tests marked @pytest.mark.dla when not on Jetson."""
    if request.node.get_closest_marker("dla"):
        try:
            from trtutils._flags import FLAGS

            if not FLAGS.IS_JETSON:
                pytest.skip("DLA tests require Jetson hardware")
        except Exception:
            pytest.skip("Cannot determine Jetson status")


# ---------------------------------------------------------------------------
# Test images
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def test_images() -> list[np.ndarray]:
    """Load all test images as numpy arrays (BGR, uint8)."""
    import cv2

    images = []
    for path in IMAGE_PATHS:
        img = cv2.imread(str(path))
        if img is not None:
            images.append(img)
    return images


@pytest.fixture(scope="session")
def horse_image() -> np.ndarray:
    """Load the horse test image."""
    import cv2

    img = cv2.imread(str(HORSE_IMAGE_PATH))
    if img is None:
        pytest.skip("Horse test image not found")
    assert img is not None
    return img


# ---------------------------------------------------------------------------
# Parametrized fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(
    params=[1, 2, 4],
    ids=["batch1", "batch2", "batch4"],
)
def batch_size(request: pytest.FixtureRequest) -> int:
    """Parametrized batch size for batch processing tests."""
    return request.param


# ---------------------------------------------------------------------------
# Factory fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def random_images() -> Callable[..., list[np.ndarray]]:
    """
    Factory fixture: generate random uint8 images.

    Usage:
        images = random_images(count=4, height=480, width=640)
    """
    rng = np.random.default_rng(42)

    def _make(
        count: int = 1,
        height: int = 480,
        width: int = 640,
        channels: int = 3,
    ) -> list[np.ndarray]:
        return [
            rng.integers(0, 255, (height, width, channels), dtype=np.uint8) for _ in range(count)
        ]

    return _make


@pytest.fixture(scope="session")
def build_test_engine() -> Callable[..., Path]:
    """
    Factory fixture: build and cache a TRT engine from an ONNX file.

    Engines are cached with TRT version in the filename so they
    are automatically rebuilt when TRT is upgraded.
    """

    def _build(
        onnx_path: Path,
        engine_dir: Path | None = None,
        optimization_level: int = 1,
    ) -> Path:
        if engine_dir is None:
            engine_dir = ENGINES_DIR
        engine_dir.mkdir(parents=True, exist_ok=True)

        engine_path = version_engine_path(engine_dir / f"{onnx_path.stem}.engine")

        if not engine_path.exists():
            from trtutils.builder import build_engine

            build_engine(
                onnx_path,
                engine_path,
                optimization_level=optimization_level,
            )

        return engine_path

    return _build
