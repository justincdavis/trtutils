# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Model test fixtures -- engines built through the model classes so their build hooks apply."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import pytest
import tensorrt as trt

from tests.conftest import ENGINES_DIR

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="session")
def build_model_engine() -> Callable[..., Path]:
    """
    Build and cache an engine via model_cls.build(), skipping if the ONNX is missing.

    Factory: build_model_engine(model_cls, onnx_path, batch_size=1). ONNX files are named <model>_<imgsz>.onnx.
    """

    def _build(model_cls: type, onnx_path: Path, batch_size: int = 1) -> Path:
        if not onnx_path.exists():
            pytest.skip(f"missing {onnx_path}")
        ENGINES_DIR.mkdir(parents=True, exist_ok=True)
        suffix = f"_b{batch_size}" if batch_size > 1 else ""
        engine_path = (
            ENGINES_DIR / f"{onnx_path.stem}_{model_cls.__name__}{suffix}_{trt.__version__}.engine"
        )
        if not engine_path.exists():
            imgsz = int(onnx_path.stem.rsplit("_", 1)[1])
            model_cls.build(onnx_path, engine_path, imgsz=imgsz, batch_size=batch_size, opt_level=1)
        return engine_path

    return _build
