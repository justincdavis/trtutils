# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/_flags.py -- FLAGS dataclass and version detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from trtutils._flags import FLAGS, Flags

ALL_FLAG_ATTRS = [
    "TRT_10",
    "TRT_HAS_UINT8",
    "TRT_HAS_INT64",
    "NEW_CAN_RUN_ON_DLA",
    "MEMSIZE_V2",
    "BUILD_PROGRESS",
    "BUILD_SERIALIZED",
    "EXEC_ASYNC_V3",
    "EXEC_ASYNC_V2",
    "EXEC_ASYNC_V1",
    "EXEC_V2",
    "EXEC_V1",
    "IS_JETSON",
    "JIT",
    "FOUND_NUMBA",
    "WARNED_NUMBA_NOT_FOUND",
    "NVTX_ENABLED",
]


@pytest.mark.cpu
def test_flags_is_flags_instance() -> None:
    """FLAGS is an instance of Flags."""
    assert isinstance(FLAGS, Flags)


@pytest.mark.cpu
@pytest.mark.parametrize("attr", ALL_FLAG_ATTRS)
def test_flag_is_bool(attr: str) -> None:
    """Each flag attribute is a bool."""
    value = getattr(FLAGS, attr)
    assert isinstance(value, bool), f"FLAGS.{attr} is {type(value).__name__}, expected bool"


@pytest.mark.cpu
def test_flag_consistency() -> None:
    """Related flags are logically consistent."""
    # EXEC_ASYNC_V3 implies EXEC_V2
    if FLAGS.EXEC_ASYNC_V3:
        assert FLAGS.EXEC_V2 is True
    # EXEC_ASYNC_V2 implies V1
    if FLAGS.EXEC_ASYNC_V2:
        assert FLAGS.EXEC_ASYNC_V1 is True
    # TRT_10 implies newer features
    if FLAGS.TRT_10:
        assert FLAGS.EXEC_ASYNC_V3 is True
        assert FLAGS.BUILD_SERIALIZED is True


@pytest.mark.cpu
def test_jetson_detection() -> None:
    """IS_JETSON matches /etc/nv_tegra_release existence."""
    tegra_exists = Path("/etc/nv_tegra_release").exists()
    assert tegra_exists == FLAGS.IS_JETSON


@pytest.mark.cpu
def test_flag_defaults() -> None:
    """NVTX_ENABLED and WARNED_NUMBA_NOT_FOUND default to False."""
    assert FLAGS.NVTX_ENABLED is False
    assert FLAGS.WARNED_NUMBA_NOT_FOUND is False
