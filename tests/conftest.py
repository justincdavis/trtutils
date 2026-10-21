# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import os

# stub injection MUST run before any trtutils imports
_CPU_ONLY = os.environ.get("TRTUTILS_IGNORE_MISSING_CUDA", "0") == "1"
if _CPU_ONLY:
    from tests._cpu_stubs import inject

    inject()

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
ENGINES_DIR = DATA_DIR / "engines"

# (min, opt, max) profiles for dynamic builds of data/simple.onnx (1x3x160x160,
# elementwise x * 0.5 + 1): batch-only, non-batch, and all dims dynamic
SIMPLE_DYNAMIC_PROFILES = {
    "batch": ((1, 3, 160, 160), (2, 3, 160, 160), (4, 3, 160, 160)),
    "spatial": ((1, 3, 32, 48), (1, 3, 96, 96), (1, 3, 160, 192)),
    "all-dims": ((1, 1, 32, 32), (2, 3, 64, 64), (3, 4, 96, 128)),
}

if not _CPU_ONLY:
    from tests._gpu_fixtures import *  # noqa: F403
