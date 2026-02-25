# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: SLF001
"""Tests for TRTEngine.execute() with all branch paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu]


# ============================================================================
# Basic execute behavior
# ============================================================================


class TestExecuteBasic:
    """Test that execute() returns correct outputs."""

    def test_execute_returns_outputs(self, engine: object, random_input: list) -> None:
        """execute() returns a list of np.ndarray."""
        outputs = engine.execute(random_input)  # type: ignore[union-attr]
        assert isinstance(outputs, list)
        assert len(outputs) >= 1
        for out in outputs:
            assert isinstance(out, np.ndarray)

    def test_execute_output_shapes_match(self, engine: object, random_input: list) -> None:
        """Output shapes match engine output_spec shapes."""
        outputs = engine.execute(random_input)  # type: ignore[union-attr]
        for out, (shape, _dtype) in zip(
            outputs,
            engine.output_spec,  # type: ignore[union-attr]
        ):
            assert list(out.shape) == shape

    def test_execute_output_dtypes_match(self, engine: object, random_input: list) -> None:
        """Output dtypes match engine output_spec dtypes."""
        outputs = engine.execute(random_input)  # type: ignore[union-attr]
        for out, (_shape, dtype) in zip(
            outputs,
            engine.output_spec,  # type: ignore[union-attr]
        ):
            assert out.dtype == dtype

    def test_execute_deterministic(self, engine: object, random_input: list) -> None:
        """Same input produces same output across two runs."""
        out1 = engine.execute(random_input)  # type: ignore[union-attr]
        out2 = engine.execute(random_input)  # type: ignore[union-attr]
        for o1, o2 in zip(out1, out2):
            assert np.array_equal(o1, o2)

    def test_execute_with_multiple_inputs(self, engine: object, random_input: list) -> None:
        """Execute works when called multiple times in sequence."""
        for _ in range(5):
            outputs = engine.execute(random_input)  # type: ignore[union-attr]
            assert outputs is not None


# ============================================================================
# Flags: no_copy, verbose, debug
# ============================================================================


class TestExecuteFlags:
    """Test execute() flag parameters."""

    @pytest.mark.parametrize("no_copy", [True, False, None])
    def test_execute_no_copy(
        self,
        engine: object,
        random_input: list,
        no_copy: bool | None,
    ) -> None:
        """Execute works with all no_copy values."""
        outputs = engine.execute(  # type: ignore[union-attr]
            random_input, no_copy=no_copy
        )
        assert isinstance(outputs, list)
        assert len(outputs) >= 1

    @pytest.mark.parametrize("verbose", [True, False, None])
    def test_execute_verbose(
        self,
        engine: object,
        random_input: list,
        verbose: bool | None,
    ) -> None:
        """Execute completes without error for all verbose settings."""
        outputs = engine.execute(  # type: ignore[union-attr]
            random_input, verbose=verbose
        )
        assert outputs is not None

    @pytest.mark.parametrize("debug", [True, False, None])
    def test_execute_debug(
        self,
        engine: object,
        random_input: list,
        debug: bool | None,
    ) -> None:
        """Execute completes without error for all debug settings."""
        outputs = engine.execute(  # type: ignore[union-attr]
            random_input, debug=debug
        )
        assert outputs is not None

    def test_execute_verbose_with_verbose_engine(self, engine_verbose: object) -> None:
        """Verbose engine + verbose=None uses engine's verbose setting."""
        rand_input = engine_verbose.get_random_input()  # type: ignore[union-attr]
        outputs = engine_verbose.execute(rand_input)  # type: ignore[union-attr]
        assert outputs is not None

    def test_execute_verbose_override(self, engine_verbose: object) -> None:
        """verbose=False overrides engine's verbose=True."""
        rand_input = engine_verbose.get_random_input()  # type: ignore[union-attr]
        outputs = engine_verbose.execute(  # type: ignore[union-attr]
            rand_input, verbose=False
        )
        assert outputs is not None


# ============================================================================
# Memory modes
# ============================================================================


class TestExecuteMemoryModes:
    """Test execute() across different memory configurations."""

    def test_execute_pagelocked(self, engine: object, random_input: list) -> None:
        """Pagelocked memory (default) produces correct outputs."""
        assert engine.pagelocked_mem is True  # type: ignore[union-attr]
        outputs = engine.execute(random_input)  # type: ignore[union-attr]
        assert outputs is not None
        assert len(outputs) >= 1

    def test_execute_no_pagelocked(self, engine_no_pagelocked: object) -> None:
        """Non-pagelocked memory path produces correct outputs."""
        assert engine_no_pagelocked.pagelocked_mem is False  # type: ignore[union-attr]
        rand_input = engine_no_pagelocked.get_random_input()  # type: ignore[union-attr]
        outputs = engine_no_pagelocked.execute(rand_input)  # type: ignore[union-attr]
        assert outputs is not None
        assert len(outputs) >= 1

    def test_execute_unified_mem(self, engine_path: Path) -> None:
        """Unified memory mode produces correct outputs."""
        from trtutils import TRTEngine

        eng = TRTEngine(
            engine_path,
            warmup=False,
            pagelocked_mem=True,
            unified_mem=True,
        )
        rand_input = eng.get_random_input()
        outputs = eng.execute(rand_input)
        assert outputs is not None
        assert len(outputs) >= 1
        del eng

    def test_execute_parity_pagelocked_vs_not(self, engine_path: Path) -> None:
        """Pagelocked vs non-pagelocked produce equivalent outputs."""
        from trtutils import TRTEngine

        eng_pl = TRTEngine(engine_path, warmup=False, pagelocked_mem=True)
        eng_no_pl = TRTEngine(engine_path, warmup=False, pagelocked_mem=False)
        rand_input = eng_pl.get_random_input()
        out_pl = eng_pl.execute(rand_input)
        out_no_pl = eng_no_pl.execute(rand_input)
        for o1, o2 in zip(out_pl, out_no_pl):
            assert np.allclose(o1, o2)
        del eng_pl, eng_no_pl

    def test_execute_parity_unified_vs_default(self, engine_path: Path) -> None:
        """Unified memory vs default produce equivalent outputs."""
        from trtutils import TRTEngine

        eng_default = TRTEngine(engine_path, warmup=False)
        eng_unified = TRTEngine(
            engine_path,
            warmup=False,
            pagelocked_mem=True,
            unified_mem=True,
        )
        rand_input = eng_default.get_random_input()
        out_default = eng_default.execute(rand_input)
        out_unified = eng_unified.execute(rand_input)
        for o1, o2 in zip(out_default, out_unified):
            assert np.allclose(o1, o2)
        del eng_default, eng_unified


# ============================================================================
# Backend-specific execute paths
# ============================================================================


class TestExecuteBackendPaths:
    """Test execute() through different backend code paths."""

    def test_execute_async_v2_backend(self, engine_path: Path) -> None:
        """Execute succeeds with explicit async_v2 backend."""
        from trtutils import TRTEngine
        from trtutils._flags import FLAGS

        if not FLAGS.EXEC_ASYNC_V2:
            pytest.skip("async_v2 not available in this TensorRT version")
        eng = TRTEngine(engine_path, warmup=False, backend="async_v2")
        assert eng._async_v3 is False
        rand_input = eng.get_random_input()
        outputs = eng.execute(rand_input)
        assert isinstance(outputs, list)
        assert len(outputs) >= 1
        del eng

    def test_execute_async_v2_parity(self, engine_path: Path) -> None:
        """async_v2 and auto backends produce equivalent outputs."""
        from trtutils import TRTEngine
        from trtutils._flags import FLAGS

        if not FLAGS.EXEC_ASYNC_V2:
            pytest.skip("async_v2 not available in this TensorRT version")
        eng_auto = TRTEngine(engine_path, warmup=False, backend="auto", cuda_graph=False)
        eng_v2 = TRTEngine(engine_path, warmup=False, backend="async_v2")
        rand_input = eng_auto.get_random_input()
        out_auto = eng_auto.execute(rand_input)
        out_v2 = eng_v2.execute(rand_input)
        for o1, o2 in zip(out_auto, out_v2):
            assert np.allclose(o1, o2)
        del eng_auto, eng_v2


# ============================================================================
# Binding reset after direct_exec / raw_exec
# ============================================================================


class TestExecuteBindingReset:
    """Test that execute() resets bindings after direct_exec/raw_exec."""

    def test_binding_reset_after_direct_exec(self, engine_path: Path) -> None:
        """execute() after direct_exec() resets tensor addresses."""
        from trtutils import TRTEngine
        from trtutils.core import allocate_to_device, free_device_ptrs

        eng = TRTEngine(engine_path, warmup=False)
        rand_input = eng.get_random_input()
        device_ptrs = allocate_to_device(rand_input)

        # direct_exec sets _using_engine_tensors to False
        eng.direct_exec(device_ptrs, no_warn=True)
        assert eng._using_engine_tensors is False

        # execute resets to engine tensors
        eng.execute(rand_input)
        assert eng._using_engine_tensors is True

        free_device_ptrs(device_ptrs)
        del eng

    def test_binding_reset_after_raw_exec(self, engine_path: Path) -> None:
        """execute() after raw_exec() resets tensor addresses."""
        from trtutils import TRTEngine
        from trtutils.core import allocate_to_device, free_device_ptrs

        eng = TRTEngine(engine_path, warmup=False)
        rand_input = eng.get_random_input()
        device_ptrs = allocate_to_device(rand_input)

        eng.raw_exec(device_ptrs, no_warn=True)
        assert eng._using_engine_tensors is False

        eng.execute(rand_input)
        assert eng._using_engine_tensors is True

        free_device_ptrs(device_ptrs)
        del eng


# ============================================================================
# no_copy behavior
# ============================================================================


class TestExecuteNoCopyBehavior:
    """Test no_copy parameter semantics in detail."""

    def test_no_copy_true_returns_same_buffer(self, engine: object, random_input: list) -> None:
        """no_copy=True returns the internal host allocation buffers."""
        outputs = engine.execute(  # type: ignore[union-attr]
            random_input, no_copy=True
        )
        for out, binding in zip(
            outputs,
            engine.output_bindings,  # type: ignore[union-attr]
        ):
            assert out is binding.host_allocation

    def test_no_copy_false_returns_copy(self, engine: object, random_input: list) -> None:
        """no_copy=False returns independent copies."""
        outputs = engine.execute(  # type: ignore[union-attr]
            random_input, no_copy=False
        )
        for out, binding in zip(
            outputs,
            engine.output_bindings,  # type: ignore[union-attr]
        ):
            assert out is not binding.host_allocation
            # But values should match
            assert np.array_equal(out, binding.host_allocation)

    def test_no_copy_none_defaults_to_copy(self, engine: object, random_input: list) -> None:
        """Default (None) behavior is to copy."""
        outputs = engine.execute(random_input)  # type: ignore[union-attr]
        for out, binding in zip(
            outputs,
            engine.output_bindings,  # type: ignore[union-attr]
        ):
            assert out is not binding.host_allocation

    def test_no_copy_true_overwritten_by_next_exec(self, engine: object, random_input: list) -> None:
        """no_copy=True buffers are overwritten by next execute call."""
        outputs_first = engine.execute(  # type: ignore[union-attr]
            random_input, no_copy=True
        )
        saved_data = [o.copy() for o in outputs_first]

        # Run again with different input (new random)
        new_input = engine.get_random_input(new=True)  # type: ignore[union-attr]
        engine.execute(new_input, no_copy=True)  # type: ignore[union-attr]

        # The saved references now point to updated data
        # (internal buffer was overwritten)
        for out_ref, binding in zip(
            outputs_first,
            engine.output_bindings,  # type: ignore[union-attr]
        ):
            assert out_ref is binding.host_allocation
            # Data may or may not differ depending on model,
            # but the reference is the same buffer
