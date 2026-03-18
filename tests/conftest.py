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

if not _CPU_ONLY:
    from tests._gpu_fixtures import *  # noqa: F403
