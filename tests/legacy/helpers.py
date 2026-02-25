# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    import numpy as np

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
ENGINES_DIR = DATA_DIR / "engines"

HORSE_IMAGE_PATH: str = str(DATA_DIR / "horse.jpg")
PEOPLE_IMAGE_PATH: str = str(DATA_DIR / "people.jpeg")
IMAGE_PATHS: list[str] = [HORSE_IMAGE_PATH, PEOPLE_IMAGE_PATH]

CLASSIFIER_ONNX_PATH = DATA_DIR / "onnx" / "resnet18.onnx"
CLASSIFIER_ENGINE_DIR = DATA_DIR / "engines" / "classifier"


def read_image(path: str | Path) -> np.ndarray:
    """
    Read an image from disk.

    Parameters
    ----------
    path : str or Path
        Path to the image file.

    Returns
    -------
    np.ndarray
        The loaded image in BGR format.

    Raises
    ------
    FileNotFoundError
        If the image could not be read.

    """
    img = cv2.imread(str(path))
    if img is None:
        err_msg = f"Failed to read image: {path}"
        raise FileNotFoundError(err_msg)
    return img
