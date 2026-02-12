# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="misc,no-any-return"
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pytest

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
HORSE_IMAGE_PATH = str(DATA_DIR / "horse.jpg")
PEOPLE_IMAGE_PATH = str(DATA_DIR / "people.jpeg")
IMAGE_PATHS = [HORSE_IMAGE_PATH, PEOPLE_IMAGE_PATH]


def _read_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        err_msg = f"Failed to read image: {path}"
        raise FileNotFoundError(err_msg)
    return img


@pytest.fixture
def test_images() -> list[np.ndarray]:
    """
    Return the set of test images as arrays.

    Returns
    -------
    list[np.ndarray]
        Loaded test images.

    """
    return [_read_image(p) for p in IMAGE_PATHS]


@pytest.fixture
def horse_image() -> np.ndarray:
    """
    Return the horse image as an array.

    Returns
    -------
    np.ndarray
        The horse image.

    """
    return _read_image(HORSE_IMAGE_PATH)


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
