# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/_flags.py -- FLAGS dataclass and version detection."""

from __future__ import annotations

import dataclasses
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
class TestFlagsDataclass:
    """Tests that FLAGS is a proper dataclass with expected structure."""

    def test_flags_is_dataclass(self) -> None:
        """FLAGS should be a dataclass instance."""
        assert dataclasses.is_dataclass(FLAGS)

    def test_flags_is_flags_instance(self) -> None:
        """FLAGS should be an instance of the Flags class."""
        assert isinstance(FLAGS, Flags)

    def test_flags_has_all_attributes(self) -> None:
        """FLAGS should have all expected attributes."""
        for attr in ALL_FLAG_ATTRS:
            assert hasattr(FLAGS, attr), f"FLAGS missing attribute: {attr}"


@pytest.mark.cpu
class TestFlagTypes:
    """Tests that each flag is the correct type (bool)."""

    @pytest.mark.parametrize("attr", ALL_FLAG_ATTRS)
    def test_flag_is_bool(self, attr) -> None:
        """Each flag attribute should be a bool."""
        value = getattr(FLAGS, attr)
        assert isinstance(value, bool), f"FLAGS.{attr} is {type(value).__name__}, expected bool"


@pytest.mark.cpu
class TestFlagConsistency:
    """Tests for logical consistency between related flags."""

    def test_exec_backend_primary_consistency(self) -> None:
        """If EXEC_ASYNC_V3 is True, execute_v2 path should be available."""
        if FLAGS.EXEC_ASYNC_V3:
            assert FLAGS.EXEC_V2 is True

    def test_exec_async_v2_implies_v1(self) -> None:
        """If EXEC_ASYNC_V2 is True, V1 should also be True."""
        if FLAGS.EXEC_ASYNC_V2:
            assert FLAGS.EXEC_ASYNC_V1 is True

    def test_trt_10_consistency(self) -> None:
        """If TRT_10 is True, certain newer features should be available."""
        if FLAGS.TRT_10:
            assert FLAGS.EXEC_ASYNC_V3 is True
            assert FLAGS.BUILD_SERIALIZED is True

    def test_jetson_detection(self) -> None:
        """IS_JETSON should match whether /etc/nv_tegra_release exists."""
        tegra_exists = Path("/etc/nv_tegra_release").exists()
        assert tegra_exists == FLAGS.IS_JETSON

    def test_nvtx_default_disabled(self) -> None:
        """NVTX_ENABLED defaults to False."""
        assert FLAGS.NVTX_ENABLED is False

    def test_warned_numba_default_false(self) -> None:
        """WARNED_NUMBA_NOT_FOUND defaults to False."""
        assert FLAGS.WARNED_NUMBA_NOT_FOUND is False
