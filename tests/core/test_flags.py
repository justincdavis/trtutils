"""Tests for src/trtutils/_flags.py -- FLAGS dataclass and version detection."""

from __future__ import annotations

import pytest


@pytest.mark.cpu
class TestFlagsDataclass:
    """Tests that FLAGS is a proper dataclass with expected structure."""

    def test_flags_is_dataclass(self) -> None:
        """FLAGS should be a dataclass instance."""
        import dataclasses

        from trtutils._flags import FLAGS

        assert dataclasses.is_dataclass(FLAGS)

    def test_flags_is_flags_instance(self) -> None:
        """FLAGS should be an instance of the Flags class."""
        from trtutils._flags import FLAGS, Flags

        assert isinstance(FLAGS, Flags)

    def test_flags_has_all_attributes(self) -> None:
        """FLAGS should have all expected attributes."""
        from trtutils._flags import FLAGS

        expected_attrs = [
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
        for attr in expected_attrs:
            assert hasattr(FLAGS, attr), f"FLAGS missing attribute: {attr}"


@pytest.mark.cpu
class TestFlagTypes:
    """Tests that each flag is the correct type (bool)."""

    @pytest.mark.parametrize(
        "attr",
        [
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
        ],
    )
    def test_flag_is_bool(self, attr) -> None:
        """Each flag attribute should be a bool."""
        from trtutils._flags import FLAGS

        value = getattr(FLAGS, attr)
        assert isinstance(value, bool), f"FLAGS.{attr} is {type(value).__name__}, expected bool"


@pytest.mark.cpu
class TestFlagConsistency:
    """Tests for logical consistency between related flags."""

    def test_exec_backend_primary_consistency(self) -> None:
        """If EXEC_ASYNC_V3 is True, execute_v2 path should be available."""
        from trtutils._flags import FLAGS

        # TRT 10 can expose V3 while not exposing legacy async entry points.
        if FLAGS.EXEC_ASYNC_V3:
            assert FLAGS.EXEC_V2 is True

    def test_exec_async_v2_implies_v1(self) -> None:
        """If EXEC_ASYNC_V2 is True, V1 should also be True."""
        from trtutils._flags import FLAGS

        if FLAGS.EXEC_ASYNC_V2:
            assert FLAGS.EXEC_ASYNC_V1 is True

    def test_trt_10_consistency(self) -> None:
        """If TRT_10 is True, certain newer features should be available."""
        from trtutils._flags import FLAGS

        if FLAGS.TRT_10:
            # TRT 10+ should have execute_async_v3
            assert FLAGS.EXEC_ASYNC_V3 is True
            # TRT 10+ should support build_serialized_network
            assert FLAGS.BUILD_SERIALIZED is True

    def test_jetson_detection(self) -> None:
        """IS_JETSON should match whether /etc/nv_tegra_release exists."""
        from pathlib import Path

        from trtutils._flags import FLAGS

        tegra_exists = Path("/etc/nv_tegra_release").exists()
        assert tegra_exists == FLAGS.IS_JETSON

    def test_internal_flags_defaults(self) -> None:
        """Internal flags JIT and FOUND_NUMBA should be bool."""
        from trtutils._flags import FLAGS

        # JIT defaults to False unless explicitly enabled
        assert isinstance(FLAGS.JIT, bool)
        assert isinstance(FLAGS.FOUND_NUMBA, bool)

    def test_nvtx_default_disabled(self) -> None:
        """NVTX_ENABLED defaults to False."""
        from trtutils._flags import FLAGS

        # NVTX is disabled by default; it must be explicitly enabled
        assert FLAGS.NVTX_ENABLED is False

    def test_warned_numba_default_false(self) -> None:
        """WARNED_NUMBA_NOT_FOUND defaults to False."""
        from trtutils._flags import FLAGS

        assert FLAGS.WARNED_NUMBA_NOT_FOUND is False


@pytest.mark.cpu
class TestFlagsImmutableBehavior:
    """Tests verifying FLAGS fields can be read reliably."""

    def test_flags_fields_are_readable(self) -> None:
        """All FLAGS fields can be read without error."""
        from trtutils._flags import FLAGS

        # Simply accessing every field should not raise
        _ = FLAGS.TRT_10
        _ = FLAGS.TRT_HAS_UINT8
        _ = FLAGS.TRT_HAS_INT64
        _ = FLAGS.NEW_CAN_RUN_ON_DLA
        _ = FLAGS.MEMSIZE_V2
        _ = FLAGS.BUILD_PROGRESS
        _ = FLAGS.BUILD_SERIALIZED
        _ = FLAGS.EXEC_ASYNC_V3
        _ = FLAGS.EXEC_ASYNC_V2
        _ = FLAGS.EXEC_ASYNC_V1
        _ = FLAGS.EXEC_V2
        _ = FLAGS.EXEC_V1
        _ = FLAGS.IS_JETSON
        _ = FLAGS.JIT
        _ = FLAGS.FOUND_NUMBA
        _ = FLAGS.WARNED_NUMBA_NOT_FOUND
        _ = FLAGS.NVTX_ENABLED

    def test_flags_consistent_across_reads(self) -> None:
        """Reading FLAGS attributes twice returns the same values."""
        from trtutils._flags import FLAGS

        assert FLAGS.TRT_10 == FLAGS.TRT_10
        assert FLAGS.EXEC_ASYNC_V3 == FLAGS.EXEC_ASYNC_V3
        assert FLAGS.IS_JETSON == FLAGS.IS_JETSON
