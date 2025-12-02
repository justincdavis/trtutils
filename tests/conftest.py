# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path

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
    return [_read_image(p) for p in IMAGE_PATHS]


@pytest.fixture
def horse_image() -> np.ndarray:
    return _read_image(HORSE_IMAGE_PATH)


@pytest.fixture(params=[1, 2, 4])
def batch_size(request) -> int:
    return request.param


@pytest.fixture
def random_images():
    def _make(num: int, height: int = 480, width: int = 640) -> list[np.ndarray]:
        return [
            np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            for _ in range(num)
        ]
    return _make
