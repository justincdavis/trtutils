# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path


def get_trt_version() -> str | None:
    """Get full TRT version string for engine filename tagging."""
    try:
        from trtutils.compat._libs import trt  # noqa: PLC0415

        return trt.__version__  # noqa: TRY300
    except (ImportError, AttributeError):
        return None


def version_engine_path(base_path: Path, trt_version: str | None = None) -> Path:
    """
    Insert TRT version into engine filename before the .engine extension.

    Example: simple.engine -> simple_10.14.1.48.post1.engine
    """
    if trt_version is None:
        return base_path
    stem = base_path.stem
    return base_path.parent / f"{stem}_{trt_version}.engine"


TRT_VERSION = get_trt_version()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
ENGINES_DIR = DATA_DIR / "engines"
ENGINE_PATH = version_engine_path(ENGINES_DIR / "simple.engine", TRT_VERSION)
ONNX_PATH = DATA_DIR / "simple.onnx"
