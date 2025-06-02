# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
ENGINES_DIR = DATA_DIR / "engines"
ENGINE_PATH = ENGINES_DIR / "simple.engine"
ONNX_PATH = DATA_DIR / "simple.onnx"
