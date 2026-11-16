# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Image test fixtures -- built engines, a CUDA stream, and device outputs for the image tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnx
import pytest
import tensorrt as trt
from onnx import TensorProto, helper

from tests._gpu_fixtures import engine_cache_path
from tests.conftest import DATA_DIR, ENGINES_DIR
from trtutils._flags import FLAGS
from trtutils.builder import build_engine
from trtutils.core import create_stream, destroy_stream
from trtutils.core._buffer import Buffer, MemoryLocation
from trtutils.models import Hands23

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

YOLOV10_ONNX = DATA_DIR / "yolov10" / "yolov10n_640.onnx"
YOLOV10_DYN_ONNX = DATA_DIR / "yolov10" / "yolov10n_640_dyn.onnx"
HANDS23_ONNX = DATA_DIR / "hands23" / "hands23_x101_800.onnx"


@pytest.fixture(scope="session")
def yolov10_engine(build_test_engine) -> Path:
    """Build and cache a YOLOv10n engine, skipping if the ONNX is missing."""
    if not YOLOV10_ONNX.exists():
        pytest.skip(f"missing {YOLOV10_ONNX}")
    return build_test_engine(YOLOV10_ONNX)


@pytest.fixture(scope="session")
def yolov10_fp16_engine() -> Path:
    """Build a YOLOv10n engine with float16 input and output, skipping if the ONNX is missing."""
    if not FLAGS.WEAK_TYPING:
        pytest.skip("float16 I/O on a float32 ONNX needs a weakly-typed network")
    if not YOLOV10_ONNX.exists():
        pytest.skip(f"missing {YOLOV10_ONNX}")
    engine_path = engine_cache_path(f"{YOLOV10_ONNX.stem}_fp16io")
    if not engine_path.exists():
        ENGINES_DIR.mkdir(parents=True, exist_ok=True)
        half = (trt.DataType.HALF, trt.TensorFormat.LINEAR)
        build_engine(
            YOLOV10_ONNX,
            engine_path,
            optimization_level=1,
            input_tensor_formats=[("images", *half)],
            output_tensor_formats=[("output0", *half)],
            fp16=True,
        )
    return engine_path


@pytest.fixture(scope="session")
def uint8_nchw_engine(tmp_path_factory, build_test_engine) -> Path:
    """Build an engine whose (1, 3, 32, 32) uint8 input is cast to float32 in the graph."""
    if not FLAGS.TRT_HAS_UINT8:
        pytest.skip("TensorRT on this platform has no UINT8 network inputs")
    onnx_path = tmp_path_factory.mktemp("uint8_nchw") / "uint8_nchw.onnx"
    shape = [1, 3, 32, 32]
    graph = helper.make_graph(
        [helper.make_node("Cast", ["input"], ["output"], to=TensorProto.FLOAT)],
        "uint8_nchw",
        [helper.make_tensor_value_info("input", TensorProto.UINT8, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(onnx_path))
    return build_test_engine(onnx_path)


@pytest.fixture(scope="session")
def yolov10_dynamic_engine() -> Path:
    """Build a YOLOv10n engine with a (1, 4, 8) batch profile, skipping if the ONNX is missing."""
    if not YOLOV10_DYN_ONNX.exists():
        pytest.skip(f"missing {YOLOV10_DYN_ONNX} (trtutils download --model yolov10n --dynamic)")
    engine_path = engine_cache_path(YOLOV10_DYN_ONNX.stem, "1-4-8")
    if not engine_path.exists():
        ENGINES_DIR.mkdir(parents=True, exist_ok=True)
        shape = (3, 640, 640)
        build_engine(
            YOLOV10_DYN_ONNX,
            engine_path,
            optimization_level=1,
            shapes=[("images", ((1, *shape), (4, *shape), (8, *shape)))],
        )
    return engine_path


@pytest.fixture(scope="session")
def task_engine(build_test_engine) -> Callable[[str], Path]:
    """Build and cache a YOLOv11n task engine ("seg", "pose", "obb", "cls"), skipping if missing."""

    def _get(task: str, batch_size: int = 1) -> Path:
        suffix = f"-b{batch_size}" if batch_size > 1 else ""
        onnx_path = DATA_DIR / "yolov11_tasks" / f"yolov11n-{task}{suffix}.onnx"
        if not onnx_path.exists():
            pytest.skip(f"missing {onnx_path} (export with `trtutils download`)")
        return build_test_engine(onnx_path, batch_size=batch_size)

    return _get


@pytest.fixture(scope="session")
def hands23_engine() -> Path:
    """Build and cache a Hands23 engine as tests/models does, skipping if the ONNX is missing."""
    if FLAGS.TRT_VERSION < (10, 0):
        pytest.skip("TensorRT 8.6 cannot build Hands23 (OneHot shape tensor)")
    if not HANDS23_ONNX.exists():
        pytest.skip(f"missing {HANDS23_ONNX}")
    engine_path = engine_cache_path(f"{HANDS23_ONNX.stem}_Hands23")
    if not engine_path.exists():
        ENGINES_DIR.mkdir(parents=True, exist_ok=True)
        Hands23.build(HANDS23_ONNX, engine_path, imgsz=800, batch_size=1, opt_level=1)
    return engine_path


@pytest.fixture
def cuda_stream():
    """Create a CUDA stream for the test, destroy after."""
    stream = create_stream()
    yield stream
    destroy_stream(stream)


@pytest.fixture
def to_device() -> Callable[[np.ndarray], Buffer]:
    """Copy a host array into a new device Buffer, as an engine output for the CUDA postprocessors."""
    return lambda array: Buffer.from_array(np.ascontiguousarray(array), MemoryLocation.DEVICE)
