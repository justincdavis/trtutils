# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="misc,no-any-return"
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import pytest

import trtutils.builder

if TYPE_CHECKING:
    from pathlib import Path

from .helpers import (
    CLASSIFIER_ENGINE_DIR,
    CLASSIFIER_ONNX_PATH,
    HORSE_IMAGE_PATH,
    IMAGE_PATHS,
    read_image,
)

CLASSIFIER_ENGINE_PATH = CLASSIFIER_ENGINE_DIR / "resnet18.engine"


@pytest.fixture
def test_images() -> list[np.ndarray]:
    """
    Return the set of test images as arrays.

    Returns
    -------
    list[np.ndarray]
        Loaded test images.

    """
    return [read_image(p) for p in IMAGE_PATHS]


@pytest.fixture
def horse_image() -> np.ndarray:
    """
    Return the horse image as an array.

    Returns
    -------
    np.ndarray
        The horse image.

    """
    return read_image(HORSE_IMAGE_PATH)


@pytest.fixture(params=[1, 2, 4])
def batch_size(request: pytest.FixtureRequest) -> int:
    """
    Provide common batch sizes for tests.

    Returns
    -------
    int
        A batch size value.

    """
    return request.param


@pytest.fixture
def random_images() -> Callable[[int, int, int], list[np.ndarray]]:
    """
    Generate random uint8 image arrays.

    Returns
    -------
    Callable[[int, int, int], list[np.ndarray]]
        Factory for random images.

    """
    rng = np.random.default_rng()

    def _make(num: int, height: int = 480, width: int = 640) -> list[np.ndarray]:
        return [rng.integers(0, 255, (height, width, 3), dtype=np.uint8) for _ in range(num)]

    return _make


@pytest.fixture(scope="session")
def classifier_engine_path() -> Path | None:
    """
    Get classifier engine path, building if needed.

    Returns
    -------
    Path | None
        The engine path if available, None if ONNX not found.

    """
    if not CLASSIFIER_ONNX_PATH.exists():
        return None

    if CLASSIFIER_ENGINE_PATH.exists():
        return CLASSIFIER_ENGINE_PATH

    CLASSIFIER_ENGINE_PATH.parent.mkdir(parents=True, exist_ok=True)

    trtutils.builder.build_engine(
        CLASSIFIER_ONNX_PATH,
        CLASSIFIER_ENGINE_PATH,
        optimization_level=1,
    )

    return CLASSIFIER_ENGINE_PATH
