# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Preprocessor test fixtures -- the photos, and preprocessors built per test case."""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import cv2
import pytest

from tests.conftest import DATA_DIR
from tests.image.preprocessors._matrix import BATCH_SIZE, DTYPES, NORMS, Case
from trtutils._flags import FLAGS
from trtutils.image.preprocessors import (
    CPUPreprocessor,
    CUDAPreprocessor,
    DLAPreprocessor,
    TRTPreprocessor,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import numpy as np

    from trtutils.image.preprocessors import ImagePreprocessor

KEPT_PREPROCESSORS = 4


@pytest.fixture(scope="session")
def frames() -> Callable[[str, int, int], np.ndarray]:
    """Load a photo from data/ resized to (height, width) with INTER_AREA, once per size."""
    loaded: dict[tuple[str, int, int], np.ndarray] = {}

    def _load(photo: str, height: int, width: int) -> np.ndarray:
        key = (photo, height, width)
        if key not in loaded:
            image = cv2.imread(str(DATA_DIR / photo))
            loaded[key] = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return loaded[key]

    return _load


@pytest.fixture(scope="session")
def build_preprocessor() -> Callable[..., ImagePreprocessor]:
    """Build a new preprocessor for a case; keyword arguments replace its constructor's."""

    def _build(case: Case, **overrides: object) -> ImagePreprocessor:
        output_range, mean, std = NORMS[case.norm]
        args = (case.output_shape, output_range, DTYPES[case.dtype])
        kwargs: dict[str, Any] = {"resize": case.resize, "mean": mean, "std": std, **overrides}
        if case.backend == "cpu":
            return CPUPreprocessor(*args, **kwargs)
        kwargs = {"cpu_resize": case.cpu_resize, **kwargs}
        if case.backend == "cuda":
            return CUDAPreprocessor(*args, **kwargs)
        if case.backend == "trt":
            if not FLAGS.TRT_HAS_UINT8:
                pytest.skip("TensorRT on this platform has no UINT8 network inputs")
            return TRTPreprocessor(*args, batch_size=BATCH_SIZE, **kwargs)
        core = int(case.backend[3:])
        if core >= FLAGS.NUM_DLA_CORES:
            pytest.skip(f"no DLA core {core}")
        return DLAPreprocessor(*args, batch_size=BATCH_SIZE, dla_core=core, **kwargs)

    return _build


@pytest.fixture(scope="class")
def preprocessors(build_preprocessor) -> Iterator[Callable[[Case], ImagePreprocessor]]:
    """Get the preprocessor of a case, shared by the consecutive cases with its configuration."""
    built: OrderedDict[tuple[object, ...], ImagePreprocessor] = OrderedDict()

    def _get(case: Case) -> ImagePreprocessor:
        key = (
            case.backend,
            case.resize,
            case.output_shape,
            case.dtype,
            case.norm,
            case.cpu_resize,
        )
        if key not in built:
            built[key] = build_preprocessor(case)
            while len(built) > KEPT_PREPROCESSORS:
                built.popitem(last=False)
        built.move_to_end(key)
        return built[key]

    yield _get
    built.clear()
