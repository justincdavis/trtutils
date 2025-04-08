# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import trtutils


def test_namespace():
    for attr in trtutils.__all__:
        assert getattr(trtutils, attr) is not None
