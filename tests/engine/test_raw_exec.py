"""Tests for TRTEngine.raw_exec() -- returns GPU pointers."""

from __future__ import annotations

import pytest


@pytest.mark.gpu
class TestRawExec:
    """Tests for TRTEngine.raw_exec()."""

    def test_raw_exec_returns_pointers(self, engine, random_input) -> None:
        """raw_exec returns a list of integer GPU pointers."""
        ptrs = [b.allocation for b in engine.input_bindings]
        result = engine.raw_exec(ptrs, no_warn=True)
        assert isinstance(result, list)
        assert all(isinstance(p, int) for p in result)
        assert len(result) == len(engine.output_bindings)

    def test_raw_exec_pointers_are_valid(self, engine, random_input) -> None:
        """Returned pointers are non-zero integers (valid GPU allocations)."""
        ptrs = [b.allocation for b in engine.input_bindings]
        result = engine.raw_exec(ptrs, no_warn=True)
        assert all(p != 0 for p in result)

    def test_raw_exec_set_pointers_true(self, engine, random_input) -> None:
        """set_pointers=True sets tensor addresses and marks _using_engine_tensors=False."""
        ptrs = [b.allocation for b in engine.input_bindings]
        engine.raw_exec(ptrs, set_pointers=True, no_warn=True)
        assert engine._using_engine_tensors is False

    def test_raw_exec_set_pointers_false(self, engine, random_input) -> None:
        """set_pointers=False skips address setting."""
        ptrs = [b.allocation for b in engine.input_bindings]
        # First call sets pointers
        engine.raw_exec(ptrs, set_pointers=True, no_warn=True)
        # Second call skips setting pointers
        result = engine.raw_exec(ptrs, set_pointers=False, no_warn=True)
        assert isinstance(result, list)

    def test_raw_exec_no_warn_true(self, engine, random_input) -> None:
        """no_warn=True suppresses the warning."""
        ptrs = [b.allocation for b in engine.input_bindings]
        # Should not produce warning
        engine.raw_exec(ptrs, no_warn=True)

    def test_raw_exec_no_warn_false(self, engine, random_input) -> None:
        """no_warn=False (default) shows the warning."""
        ptrs = [b.allocation for b in engine.input_bindings]
        engine.raw_exec(ptrs, no_warn=False)

    def test_raw_exec_no_warn_none(self, engine, random_input) -> None:
        """no_warn=None (default) shows the warning (same as False)."""
        ptrs = [b.allocation for b in engine.input_bindings]
        engine.raw_exec(ptrs, no_warn=None)

    def test_raw_exec_debug(self, engine, random_input) -> None:
        """debug=True adds stream_synchronize after execution."""
        ptrs = [b.allocation for b in engine.input_bindings]
        result = engine.raw_exec(ptrs, no_warn=True, debug=True)
        assert isinstance(result, list)

    def test_raw_exec_verbose(self, engine_verbose, random_input) -> None:
        """Verbose engine exercises the verbose path."""
        ptrs = [b.allocation for b in engine_verbose.input_bindings]
        result = engine_verbose.raw_exec(ptrs, no_warn=True)
        assert isinstance(result, list)

    def test_raw_exec_output_pointers_match_bindings(self, engine, random_input: object) -> None:
        """Returned pointers match the output binding allocations."""
        ptrs = [b.allocation for b in engine.input_bindings]
        result = engine.raw_exec(ptrs, no_warn=True)
        expected = [b.allocation for b in engine.output_bindings]
        assert result == expected
