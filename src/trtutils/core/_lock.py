# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from threading import Lock

MEM_ALLOC_LOCK = Lock()
NVRTC_LOCK = Lock()
