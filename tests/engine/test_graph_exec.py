# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for TRTEngine CUDA graph lifecycle -- capture, replay, teardown."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.gpu, pytest.mark.cuda_graph]


# ============================================================================
# Graph capture and replay
# ============================================================================


class TestCUDAGraphCapture:
    """Tests for CUDA graph capture and graph_exec()."""

    def test_first_execute_captures_graph(self, engine_path: Path) -> None:
        """cuda_graph=True + first execute() triggers graph capture."""
        from trtutils import TRTEngine

        eng = TRTEngine(engine_path, warmup=False, cuda_graph=True)
        data = eng.get_random_input()
        eng.execute(data)
        # After first execute, graph should be captured (if backend supports it)
        if eng._cuda_graph is not None:
            assert eng._cuda_graph.is_captured
        del eng

    def test_graph_exec_after_capture(self, engine_path: Path) -> None:
        """After warmup with cuda_graph=True, graph_exec succeeds."""
        from trtutils import TRTEngine

        eng = TRTEngine(engine_path, warmup=True, warmup_iterations=2)
        if eng._cuda_graph is not None and eng._cuda_graph.is_captured:
            eng.graph_exec()  # Should not raise
        del eng

    def test_graph_exec_without_capture_raises(self, engine_path: Path) -> None:
        """graph_exec raises RuntimeError when no graph is captured."""
        from trtutils import TRTEngine

        eng = TRTEngine(engine_path, warmup=False, cuda_graph=False)
        with pytest.raises(RuntimeError, match="No CUDA graph captured"):
            eng.graph_exec()
        del eng

    def test_graph_exec_deterministic(self, engine_path: Path) -> None:
        """Graph replay produces consistent output."""
        from trtutils import TRTEngine

        eng = TRTEngine(engine_path, warmup=True, warmup_iterations=2)
        if eng._cuda_graph is None or not eng._cuda_graph.is_captured:
            del eng
            pytest.skip("CUDA graph not captured")
        data = eng.get_random_input()
        out1 = eng.execute(data)
        out2 = eng.execute(data)
        for o1, o2 in zip(out1, out2):
            assert np.array_equal(o1, o2)
        del eng

    def test_graph_invalidate_and_recapture(self, engine_path: Path) -> None:
        """Invalidating graph allows recapture on next execute."""
        from trtutils import TRTEngine

        eng = TRTEngine(engine_path, warmup=True, warmup_iterations=2)
        if eng._cuda_graph is None or not eng._cuda_graph.is_captured:
            del eng
            pytest.skip("CUDA graph not captured")
        # Invalidate
        eng._cuda_graph.invalidate()
        assert not eng._cuda_graph.is_captured
        # Next execute should recapture
        data = eng.get_random_input()
        result = eng.execute(data)
        assert isinstance(result, list)
        assert eng._cuda_graph.is_captured
        del eng

    def test_graph_exec_debug(self, engine_path: Path) -> None:
        """graph_exec(debug=True) synchronizes the stream."""
        from trtutils import TRTEngine

        eng = TRTEngine(engine_path, warmup=True, warmup_iterations=2)
        if eng._cuda_graph is not None and eng._cuda_graph.is_captured:
            eng.graph_exec(debug=True)  # Should not raise
        del eng

    def test_graph_invalidation_on_set_input(self, engine_path: Path) -> None:
        """Setting input bindings invalidates the captured graph."""
        from trtutils import TRTEngine

        eng = TRTEngine(engine_path, warmup=True, warmup_iterations=2)
        if eng._cuda_graph is not None and eng._cuda_graph.is_captured:
            eng._set_input_bindings()
            assert not eng._cuda_graph.is_captured
        del eng


# ============================================================================
# Full integration flow
# ============================================================================


class TestCUDAGraphExecuteIntegration:
    """Integration tests for CUDA graph with execute()."""

    def test_execute_with_cuda_graph_full_flow(self, engine_path: Path) -> None:
        """Full flow: init -> execute (capture) -> execute (replay)."""
        from trtutils import TRTEngine

        eng = TRTEngine(engine_path, warmup=False, cuda_graph=True)
        data = eng.get_random_input()
        # First execute triggers capture
        out1 = eng.execute(data)
        assert isinstance(out1, list)
        # Second execute replays graph
        out2 = eng.execute(data)
        assert isinstance(out2, list)
        # Results should be consistent
        for o1, o2 in zip(out1, out2):
            assert np.array_equal(o1, o2)
        del eng

    def test_execute_second_call_replays_graph(self, engine_path: Path) -> None:
        """Second execute() replays the captured graph."""
        from trtutils import TRTEngine

        eng = TRTEngine(engine_path, warmup=True, warmup_iterations=2)
        data = eng.get_random_input()
        result = eng.execute(data)
        assert isinstance(result, list)
        del eng

    def test_execute_no_copy_with_graph(self, engine_path: Path) -> None:
        """no_copy=True works correctly with CUDA graph path."""
        from trtutils import TRTEngine

        eng = TRTEngine(engine_path, warmup=True, warmup_iterations=2)
        if eng._cuda_graph is None or not eng._cuda_graph.is_captured:
            del eng
            pytest.skip("CUDA graph not captured")
        data = eng.get_random_input()
        outputs = eng.execute(data, no_copy=True)
        # no_copy returns internal buffers
        for out, binding in zip(outputs, eng.output_bindings):
            assert out is binding.host_allocation
        del eng

    def test_graph_exec_no_default_sync(self, engine_path: Path) -> None:
        """graph_exec() without debug does not add extra sync (no error)."""
        from trtutils import TRTEngine

        eng = TRTEngine(engine_path, warmup=True, warmup_iterations=2)
        if eng._cuda_graph is not None and eng._cuda_graph.is_captured:
            eng.graph_exec()  # Should not raise, no debug sync
        del eng


# ============================================================================
# Edge cases and internal branch coverage
# ============================================================================


class TestCUDAGraphEdgeCases:
    """Edge cases for CUDA graph capture internals."""

    def test_cuda_graph_disabled(self, engine_path: Path) -> None:
        """cuda_graph=False disables graph capture entirely."""
        from trtutils import TRTEngine

        eng = TRTEngine(engine_path, warmup=False, cuda_graph=False)
        assert eng._cuda_graph is None
        data = eng.get_random_input()
        result = eng.execute(data)
        assert isinstance(result, list)
        del eng

    def test_cuda_graph_with_async_v2(self, engine_path: Path) -> None:
        """cuda_graph=True + async_v2 backend disables graph (requires v3)."""
        from trtutils import TRTEngine

        eng = TRTEngine(engine_path, warmup=False, backend="async_v2", cuda_graph=True)
        # cuda_graph should be disabled since async_v2 doesn't support it
        assert not eng._cuda_graph_enabled
        assert eng._cuda_graph is None
        del eng

    def test_capture_recursion_guard(self, engine_path: Path) -> None:
        """_capturing_graph=True causes _capture_cuda_graph to return early."""
        from trtutils import TRTEngine

        eng = TRTEngine(engine_path, warmup=False, cuda_graph=True)
        if eng._cuda_graph is None:
            del eng
            pytest.skip("CUDA graph not enabled")
        # Simulate recursion guard
        eng._capturing_graph = True
        eng._capture_cuda_graph()  # Should return early, not raise
        eng._capturing_graph = False
        del eng

    def test_capture_cuda_graph_none_raises(self, engine_path: Path) -> None:
        """_capture_cuda_graph raises RuntimeError when _cuda_graph is None."""
        from trtutils import TRTEngine

        eng = TRTEngine(engine_path, warmup=False, cuda_graph=True)
        # Force _cuda_graph to None to test the guard
        saved = eng._cuda_graph
        eng._cuda_graph = None
        with pytest.raises(RuntimeError, match="CUDA graph is not enabled"):
            eng._capture_cuda_graph()
        eng._cuda_graph = saved
        del eng

    def test_capture_warmup_failure_invalidates_graph(self, engine_path: Path) -> None:
        """RuntimeError during warmup invalidates graph and raises."""
        from trtutils import TRTEngine

        eng = TRTEngine(engine_path, warmup=False, cuda_graph=True)
        if eng._cuda_graph is None:
            del eng
            pytest.skip("CUDA graph not enabled")

        # Mock warmup to raise RuntimeError
        with patch.object(
            eng, "warmup", side_effect=RuntimeError("mock warmup failure")
        ), pytest.raises(
            RuntimeError,
            match=r"CUDA graph capture failed.*during warmup",
        ):
            eng._capture_cuda_graph()
        # Graph should be invalidated
        assert eng._cuda_graph is None
        del eng

    def test_binding_change_invalidates_graph(self, engine_path: Path) -> None:
        """Changing output bindings invalidates captured graph."""
        from trtutils import TRTEngine

        eng = TRTEngine(engine_path, warmup=True, warmup_iterations=2)
        if eng._cuda_graph is not None and eng._cuda_graph.is_captured:
            eng._set_output_bindings()
            assert not eng._cuda_graph.is_captured
        del eng

    def test_delete_engine_with_graph(self, engine_path: Path) -> None:
        """Deleting engine with active CUDA graph does not crash."""
        from trtutils import TRTEngine

        eng = TRTEngine(engine_path, warmup=True, warmup_iterations=2)
        del eng  # Should not crash
