# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import onnx
import pytest
import tensorrt as trt

from trtutils._flags import FLAGS
from trtutils.builder import build_engine as _build_engine
from trtutils.core import get_compute_capability
from trtutils.core._engine import create_engine
from trtutils.core._stream import destroy_stream

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
ENGINES_DIR = DATA_DIR / "engines"


@dataclass(frozen=True)
class TestImage:
    """A test image with its path, pixel data, and ground truth."""

    path: Path
    array: np.ndarray
    gt_det_classes: list[int] = field(default_factory=list)
    gt_det_min: int = 0
    gt_det_max: int = 0
    gt_cls_id: int | None = None


TRT_VERSION = trt.__version__


def version_engine_path(path: Path) -> Path:
    """Insert TRT version into engine filename for caching."""
    return path.with_name(f"{path.stem}_{TRT_VERSION}{path.suffix}")


@pytest.fixture(autouse=True)
def _skip_jetson_tests(request: pytest.FixtureRequest) -> None:
    if request.node.get_closest_marker("jetson") and not FLAGS.IS_JETSON:
        pytest.skip("Jetson tests require Jetson hardware")


@pytest.fixture(scope="session")
def images() -> dict[str, TestImage]:
    """Keyed test images with ground truth. Access via images["horse"], images["people"]."""
    result: dict[str, TestImage] = {}

    horse_path = DATA_DIR / "horse.jpg"
    img = cv2.imread(str(horse_path))
    if img is not None:
        result["horse"] = TestImage(
            path=horse_path,
            array=img,
            gt_det_classes=[17],
            gt_det_min=1,
            gt_det_max=2,
            gt_cls_id=603,
        )

    people_path = DATA_DIR / "people.jpeg"
    img = cv2.imread(str(people_path))
    if img is not None:
        result["people"] = TestImage(
            path=people_path,
            array=img,
            gt_det_classes=[0],
            gt_det_min=3,
            gt_det_max=5,
        )

    return result


@pytest.fixture(params=[1, 2, 4], ids=["batch1", "batch2", "batch4"])
def batch_size(request: pytest.FixtureRequest) -> int:
    """Parametrized batch sizes."""
    return request.param


@pytest.fixture(scope="session")
def random_images() -> Callable[..., list[np.ndarray]]:
    """
    Generate random images.

    Factory: random_images(count=4, height=480, width=640).
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
    Build and cache a TRT engine from an ONNX file.

    Factory: build_test_engine(onnx_path, engine_dir=None, optimization_level=1, batch_size=1).
    """

    def _build(
        onnx_path: Path,
        engine_dir: Path | None = None,
        optimization_level: int = 1,
        batch_size: int = 1,
    ) -> Path:
        if engine_dir is None:
            engine_dir = ENGINES_DIR
        engine_dir.mkdir(parents=True, exist_ok=True)

        engine_path = version_engine_path(engine_dir / f"{onnx_path.stem}_b{batch_size}.engine")

        if not engine_path.exists():
            shapes = None
            if batch_size > 1:
                model = onnx.load(str(onnx_path))
                shapes = []
                for tensor in model.graph.input:
                    dims = []
                    for dim in tensor.type.tensor_type.shape.dim:
                        if dim.dim_param:
                            dims.append(1)
                        elif dim.dim_value:
                            dims.append(int(dim.dim_value))
                    if not dims:
                        continue
                    if len(dims) >= 1:
                        if dims[0] not in (0, 1, batch_size):
                            err_msg = f"Model {onnx_path.name} has fixed batch {dims[0]} and cannot use {batch_size}"
                            raise ValueError(err_msg)
                        dims[0] = batch_size
                    shapes.append((tensor.name, tuple(dims)))

            try:
                _build_engine(
                    onnx_path,
                    engine_path,
                    optimization_level=optimization_level,
                    shapes=shapes,
                )
            except RuntimeError:
                if not FLAGS.TRT_10:
                    sm = get_compute_capability()
                    if sm >= (8, 9):
                        pytest.skip(f"TRT <10 does not support SM {sm[0]}.{sm[1]}")
                raise

        return engine_path

    return _build


@pytest.fixture(scope="session")
def simple_onnx_path() -> Path:
    """Path to a minimal ONNX model for core tests."""
    return Path(__file__).parent.parent / "data" / "simple.onnx"


@pytest.fixture(scope="session")
def simple_engine_path(build_test_engine, simple_onnx_path) -> Path:
    """Build and return path to a simple test engine."""
    return build_test_engine(simple_onnx_path)


@pytest.fixture
def simple_engine(simple_engine_path):
    """Load simple test engine, destroy stream after test."""
    engine, context, _logger, stream = create_engine(simple_engine_path)
    yield engine, context, stream
    destroy_stream(stream)
