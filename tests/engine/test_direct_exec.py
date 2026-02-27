# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for TRTEngine.direct_exec() method paths."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.gpu]


class TestDirectExec:
    """Test direct_exec() with GPU memory pointers."""

    def test_direct_exec_returns_outputs(self, engine_path: Path) -> None:
        """direct_exec() returns list of np.ndarray."""
        from trtutils import TRTEngine
        from trtutils.core import allocate_to_device, free_device_ptrs

        eng = TRTEngine(engine_path, warmup=False)
        rand_input = eng.get_random_input()
        device_ptrs = allocate_to_device(rand_input)

        outputs = eng.direct_exec(device_ptrs, no_warn=True)
        assert isinstance(outputs, list)
        assert len(outputs) >= 1
        for out in outputs:
            assert isinstance(out, np.ndarray)

        free_device_ptrs(device_ptrs)
        del eng

    def test_direct_exec_matches_execute(self, engine_path: Path) -> None:
        """direct_exec and execute produce same results for same input."""
        from trtutils import TRTEngine
        from trtutils.core import allocate_to_device, free_device_ptrs

        eng = TRTEngine(engine_path, warmup=False)
        rand_input = eng.get_random_input()
        device_ptrs = allocate_to_device(rand_input)

        outputs_direct = eng.direct_exec(device_ptrs, no_warn=True)
        outputs_direct_copy = [o.copy() for o in outputs_direct]

        outputs_execute = eng.execute(rand_input)

        for od, oe in zip(outputs_direct_copy, outputs_execute):
            assert np.array_equal(od, oe)

        free_device_ptrs(device_ptrs)
        del eng

    @pytest.mark.parametrize("set_pointers", [True, False])
    def test_set_pointers_flag(self, engine_path: Path, set_pointers: bool) -> None:
        """Both set_pointers=True and set_pointers=False paths work."""
        from trtutils import FLAGS, TRTEngine
        from trtutils.core import allocate_to_device, free_device_ptrs

        eng = TRTEngine(engine_path, warmup=False)
        rand_input = eng.get_random_input()
        device_ptrs = allocate_to_device(rand_input)

        if not set_pointers and not FLAGS.EXEC_ASYNC_V3:
            # For async_v2, set_pointers is not applicable the same way
            # The pointers list is directly used in execute_async_v2
            pass

        if set_pointers:
            outputs = eng.direct_exec(device_ptrs, set_pointers=True, no_warn=True)
        else:
            # First call with set_pointers=True to configure addresses
            eng.direct_exec(device_ptrs, set_pointers=True, no_warn=True)
            # Second call with set_pointers=False (addresses already set)
            outputs = eng.direct_exec(device_ptrs, set_pointers=False, no_warn=True)

        assert outputs is not None
        assert len(outputs) >= 1

        free_device_ptrs(device_ptrs)
        del eng

    @pytest.mark.parametrize("no_warn", [True, False, None])
    def test_no_warn_flag(self, engine_path: Path, no_warn: bool | None) -> None:
        """Warning suppression flag works for all values."""
        from trtutils import TRTEngine
        from trtutils.core import allocate_to_device, free_device_ptrs

        eng = TRTEngine(engine_path, warmup=False)
        rand_input = eng.get_random_input()
        device_ptrs = allocate_to_device(rand_input)

        outputs = eng.direct_exec(device_ptrs, no_warn=no_warn)
        assert outputs is not None

        free_device_ptrs(device_ptrs)
        del eng

    def test_direct_exec_sets_using_engine_false(self, engine_path: Path) -> None:
        """direct_exec() with set_pointers marks _using_engine_tensors=False."""
        from trtutils import FLAGS, TRTEngine
        from trtutils.core import allocate_to_device, free_device_ptrs

        if not FLAGS.EXEC_ASYNC_V3:
            pytest.skip("Only relevant for async_v3 backend")

        eng = TRTEngine(engine_path, warmup=False, backend="async_v3")
        assert eng._using_engine_tensors is True

        rand_input = eng.get_random_input()
        device_ptrs = allocate_to_device(rand_input)

        eng.direct_exec(device_ptrs, set_pointers=True, no_warn=True)
        assert eng._using_engine_tensors is False

        free_device_ptrs(device_ptrs)
        del eng

    def test_verbose_logging(self, engine_path: Path) -> None:
        """direct_exec with verbose=True does not error."""
        from trtutils import TRTEngine
        from trtutils.core import allocate_to_device, free_device_ptrs

        eng = TRTEngine(engine_path, warmup=False, verbose=True)
        rand_input = eng.get_random_input()
        device_ptrs = allocate_to_device(rand_input)

        outputs = eng.direct_exec(device_ptrs, no_warn=True, verbose=True)
        assert outputs is not None

        free_device_ptrs(device_ptrs)
        del eng

    def test_debug_synchronization(self, engine_path: Path) -> None:
        """direct_exec with debug=True adds extra stream sync."""
        from trtutils import TRTEngine
        from trtutils.core import allocate_to_device, free_device_ptrs

        eng = TRTEngine(engine_path, warmup=False)
        rand_input = eng.get_random_input()
        device_ptrs = allocate_to_device(rand_input)

        outputs = eng.direct_exec(device_ptrs, no_warn=True, debug=True)
        assert outputs is not None

        free_device_ptrs(device_ptrs)
        del eng

    def test_direct_exec_verbose_none_uses_engine_verbose(self, engine_path: Path) -> None:
        """verbose=None falls back to engine's _verbose setting."""
        from trtutils import TRTEngine
        from trtutils.core import allocate_to_device, free_device_ptrs

        eng = TRTEngine(engine_path, warmup=False, verbose=True)
        rand_input = eng.get_random_input()
        device_ptrs = allocate_to_device(rand_input)

        # verbose=None should use engine._verbose=True
        outputs = eng.direct_exec(device_ptrs, no_warn=True, verbose=None)
        assert outputs is not None

        free_device_ptrs(device_ptrs)
        del eng


class TestDirectExecMemoryModes:
    """Test direct_exec() output copy paths for different memory modes."""

    def test_direct_exec_pagelocked(self, engine_path: Path) -> None:
        """Pagelocked memory output copy path works."""
        from trtutils import TRTEngine
        from trtutils.core import allocate_to_device, free_device_ptrs

        eng = TRTEngine(engine_path, warmup=False, pagelocked_mem=True)
        rand_input = eng.get_random_input()
        device_ptrs = allocate_to_device(rand_input)

        outputs = eng.direct_exec(device_ptrs, no_warn=True)
        assert outputs is not None
        assert len(outputs) >= 1

        free_device_ptrs(device_ptrs)
        del eng

    def test_direct_exec_no_pagelocked(self, engine_path: Path) -> None:
        """Non-pagelocked memory output copy path works."""
        from trtutils import TRTEngine
        from trtutils.core import allocate_to_device, free_device_ptrs

        eng = TRTEngine(engine_path, warmup=False, pagelocked_mem=False)
        rand_input = eng.get_random_input()
        device_ptrs = allocate_to_device(rand_input)

        outputs = eng.direct_exec(device_ptrs, no_warn=True)
        assert outputs is not None
        assert len(outputs) >= 1

        free_device_ptrs(device_ptrs)
        del eng

    def test_direct_exec_unified_mem(self, engine_path: Path) -> None:
        """Unified memory output copy path works (no-op copy)."""
        from trtutils import TRTEngine
        from trtutils.core import allocate_to_device, free_device_ptrs

        eng = TRTEngine(
            engine_path,
            warmup=False,
            pagelocked_mem=True,
            unified_mem=True,
        )
        rand_input = eng.get_random_input()
        device_ptrs = allocate_to_device(rand_input)

        outputs = eng.direct_exec(device_ptrs, no_warn=True)
        assert outputs is not None
        assert len(outputs) >= 1

        free_device_ptrs(device_ptrs)
        del eng
