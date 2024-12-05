# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

# initialize CUDA (if able)
import contextlib

with contextlib.suppress(ImportError, RuntimeError):
    from trtutils.core import create_stream

    STREAM = create_stream()
