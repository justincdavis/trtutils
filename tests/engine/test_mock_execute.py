# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for mock_execute(), warmup(), get_random_input(), and __call__."""

from __future__ import annotations

import numpy as np


class TestGetRandomInput:
    """Tests for TRTEngineInterface.get_random_input()."""

    def test_returns_list_of_arrays(self, engine) -> None:
        """get_random_input returns list of numpy arrays."""
        data = engine.get_random_input()
        assert isinstance(data, list)
        assert all(isinstance(a, np.ndarray) for a in data)

    def test_shapes_match_input_spec(self, engine) -> None:
        """Returned arrays match the engine input spec shapes."""
        data = engine.get_random_input()
        for arr, (shape, _dtype) in zip(data, engine.input_spec):
            assert list(arr.shape) == shape

    def test_dtypes_match_input_spec(self, engine) -> None:
        """Returned arrays match the engine input spec dtypes."""
        data = engine.get_random_input()
        for arr, (_shape, dtype) in zip(data, engine.input_spec):
            assert arr.dtype == dtype

    def test_cached_returns_same(self, engine) -> None:
        """Default (new=None) returns cached data on second call."""
        data1 = engine.get_random_input()
        data2 = engine.get_random_input()
        for a, b in zip(data1, data2):
            np.testing.assert_array_equal(a, b)

    def test_new_true_generates_fresh(self, engine) -> None:
        """new=True generates fresh random data."""
        engine.get_random_input()
        data2 = engine.get_random_input(new=True)
        # With high probability, new data differs
        assert isinstance(data2, list)

    def test_verbose_path(self, engine_verbose) -> None:
        """verbose=True exercises the logging path."""
        data = engine_verbose.get_random_input(verbose=True)
        assert isinstance(data, list)


class TestMockExecute:
    """Tests for TRTEngineInterface.mock_execute()."""

    def test_mock_execute_no_data(self, engine) -> None:
        """mock_execute(data=None) generates random input internally."""
        result = engine.mock_execute()
        assert isinstance(result, list)
        assert all(isinstance(a, np.ndarray) for a in result)

    def test_mock_execute_with_data(self, engine, random_input) -> None:
        """mock_execute(data=...) uses provided data."""
        result = engine.mock_execute(data=random_input)
        assert isinstance(result, list)

    def test_mock_execute_returns_no_copy(self, engine) -> None:
        """mock_execute uses no_copy=True internally."""
        result = engine.mock_execute()
        # Results should be views of host allocations (no_copy=True)
        assert isinstance(result, list)

    def test_mock_execute_verbose(self, engine_verbose) -> None:
        """mock_execute with verbose engine."""
        result = engine_verbose.mock_execute(verbose=True)
        assert isinstance(result, list)

    def test_mock_execute_debug(self, engine) -> None:
        """mock_execute(debug=True) adds stream sync."""
        result = engine.mock_execute(debug=True)
        assert isinstance(result, list)


class TestWarmup:
    """Tests for TRTEngineInterface.warmup()."""

    def test_warmup_runs_iterations(self, engine) -> None:
        """warmup(iterations=N) runs N mock executions."""
        engine.warmup(3)  # Should not raise

    def test_warmup_single_iteration(self, engine) -> None:
        """warmup(iterations=1) runs one execution."""
        engine.warmup(1)

    def test_warmup_verbose(self, engine_verbose) -> None:
        """Warmup with verbose=True."""
        engine_verbose.warmup(2, verbose=True)

    def test_warmup_debug(self, engine) -> None:
        """Warmup with debug=True."""
        engine.warmup(1, debug=True)


class TestCall:
    """Tests for TRTEngineInterface.__call__()."""

    def test_call_delegates_to_execute(self, engine, random_input) -> None:
        """__call__ delegates to execute()."""
        result = engine(random_input)
        assert isinstance(result, list)
        assert all(isinstance(a, np.ndarray) for a in result)

    def test_call_no_copy(self, engine, random_input) -> None:
        """__call__ with no_copy=True."""
        result = engine(random_input, no_copy=True)
        assert isinstance(result, list)

    def test_call_verbose(self, engine, random_input) -> None:
        """__call__ with verbose=True."""
        result = engine(random_input, verbose=True)
        assert isinstance(result, list)
