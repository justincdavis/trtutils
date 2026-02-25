# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import pytest

import trtutils


@pytest.fixture(autouse=True)
def _skip_if_not_jetson() -> None:
    if not trtutils.FLAGS.IS_JETSON:
        pytest.skip("DLA tests require Jetson hardware")
