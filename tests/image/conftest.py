# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Image test fixtures -- a built YOLOv10 engine shared by the image model tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import onnx
import pytest
import tensorrt as trt

from tests.conftest import DATA_DIR, ENGINES_DIR
from trtutils import TRTEngine
from trtutils.builder import build_engine

if TYPE_CHECKING:
    from pathlib import Path

YOLOV10_ONNX = DATA_DIR / "yolov10" / "yolov10n_640.onnx"


@pytest.fixture(scope="session")
def yolov10_engine(build_test_engine) -> Path:
    """Build and cache a YOLOv10n engine, skipping if the ONNX is missing."""
    if not YOLOV10_ONNX.exists():
        pytest.skip(f"missing {YOLOV10_ONNX}")
    return build_test_engine(YOLOV10_ONNX)


@pytest.fixture(scope="session")
def yolov10_dynamic_engine() -> Path:
    """
    Build a YOLOv10n engine with a (1, 4, 8) dynamic batch profile.

    Same technique as tests/engine/conftest.py::_build_dynamic_test_engine:
    the exported ONNX is static, so the batch dim is made symbolic on the
    input and output in place. Skips (reporting why) if the build fails, or
    if the resulting engine collapses back to a static batch -- some export
    graphs bake the batch dimension into internal ops even when the I/O
    tensors are marked dynamic.
    """
    if not YOLOV10_ONNX.exists():
        pytest.skip(f"missing {YOLOV10_ONNX}")

    engine_path = ENGINES_DIR / f"yolov10n_640_dyn_b8_{trt.__version__}.engine"
    if not engine_path.exists():
        ENGINES_DIR.mkdir(parents=True, exist_ok=True)
        model = onnx.load(str(YOLOV10_ONNX))
        for tensor in (*model.graph.input, *model.graph.output):
            dim = tensor.type.tensor_type.shape.dim[0]
            dim.ClearField("dim_value")
            dim.dim_param = "batch"
        dyn_onnx = ENGINES_DIR / "yolov10n_640_dyn.onnx"
        onnx.save(model, str(dyn_onnx))

        shape = tuple(d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim[1:])
        try:
            build_engine(
                dyn_onnx,
                engine_path,
                optimization_level=1,
                shapes=[
                    (
                        model.graph.input[0].name,
                        ((1, *shape), (4, *shape), (8, *shape)),
                    )
                ],
            )
        except Exception as e:
            pytest.skip(f"yolov10n_640 ONNX cannot be built with a dynamic batch profile: {e}")

    engine = TRTEngine(engine_path, warmup=False)
    is_dynamic = engine.is_dynamic_batch
    del engine
    if not is_dynamic:
        pytest.skip(
            "yolov10n_640 ONNX collapses to a static batch even with a symbolic "
            "input/output and optimization profile; some internal op bakes the "
            "batch dimension, so a genuine dynamic-batch engine cannot be built "
            "from it with this technique."
        )
    return engine_path
