# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/core/_graph.py -- CUDA graph capture and replay."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# CUDAGraph class lifecycle tests
# ---------------------------------------------------------------------------
@pytest.mark.gpu
class TestCUDAGraphLifecycle:
    """Tests for CUDAGraph class init, capture, launch, and invalidate."""

    def test_init_not_captured(self, cuda_stream) -> None:
        """After init, is_captured should be False."""
        from trtutils.core._graph import CUDAGraph

        graph = CUDAGraph(cuda_stream)
        assert graph.is_captured is False

    def test_context_manager_capture(self, cuda_stream) -> None:
        """Context manager start/stop marks graph as captured."""
        from trtutils.core._graph import CUDAGraph

        graph = CUDAGraph(cuda_stream)
        with graph:
            pass  # Empty capture -- captures an empty graph
        assert graph.is_captured is True
        graph.invalidate()

    def test_start_stop_capture(self, cuda_stream) -> None:
        """Manual start() and stop() marks graph as captured."""
        from trtutils.core._graph import CUDAGraph

        graph = CUDAGraph(cuda_stream)
        graph.start()
        success = graph.stop()
        assert success is True
        assert graph.is_captured is True
        graph.invalidate()

    def test_launch_after_capture(self, cuda_stream) -> None:
        """launch() succeeds after a successful capture."""
        from trtutils.core._graph import CUDAGraph
        from trtutils.core._stream import stream_synchronize

        graph = CUDAGraph(cuda_stream)
        with graph:
            pass  # Empty capture
        graph.launch()
        stream_synchronize(cuda_stream)
        graph.invalidate()

    def test_invalidate_clears_state(self, cuda_stream) -> None:
        """invalidate() sets is_captured back to False."""
        from trtutils.core._graph import CUDAGraph

        graph = CUDAGraph(cuda_stream)
        with graph:
            pass
        assert graph.is_captured is True
        graph.invalidate()
        assert graph.is_captured is False

    def test_launch_without_capture_raises(self, cuda_stream) -> None:
        """launch() without capture raises RuntimeError."""
        from trtutils.core._graph import CUDAGraph

        graph = CUDAGraph(cuda_stream)
        with pytest.raises(RuntimeError, match="no graph has been captured"):
            graph.launch()

    def test_invalidate_without_capture(self, cuda_stream) -> None:
        """invalidate() on a fresh graph does not raise."""
        from trtutils.core._graph import CUDAGraph

        graph = CUDAGraph(cuda_stream)
        graph.invalidate()  # Should not raise

    def test_invalidate_idempotent(self, cuda_stream) -> None:
        """invalidate() can be called multiple times safely."""
        from trtutils.core._graph import CUDAGraph

        graph = CUDAGraph(cuda_stream)
        with graph:
            pass
        graph.invalidate()
        graph.invalidate()  # Second call should not raise

    def test_del_calls_invalidate(self, cuda_stream) -> None:
        """__del__ cleans up graph resources without error."""
        from trtutils.core._graph import CUDAGraph

        graph = CUDAGraph(cuda_stream)
        with graph:
            pass
        assert graph.is_captured is True
        del graph  # Should not crash

    def test_context_manager_returns_self(self, cuda_stream) -> None:
        """__enter__ returns the CUDAGraph instance itself."""
        from trtutils.core._graph import CUDAGraph

        graph = CUDAGraph(cuda_stream)
        with graph as g:
            assert g is graph
        graph.invalidate()

    def test_multiple_capture_cycles(self, cuda_stream) -> None:
        """A graph can be invalidated and recaptured."""
        from trtutils.core._graph import CUDAGraph

        graph = CUDAGraph(cuda_stream)

        # First capture
        with graph:
            pass
        assert graph.is_captured is True
        graph.invalidate()
        assert graph.is_captured is False

        # Second capture
        with graph:
            pass
        assert graph.is_captured is True
        graph.invalidate()


# ---------------------------------------------------------------------------
# Low-level graph function tests
# ---------------------------------------------------------------------------
@pytest.mark.gpu
class TestCUDAGraphFunctions:
    """Tests for the low-level CUDA graph functions."""

    def test_begin_end_capture(self, cuda_stream) -> None:
        """cuda_stream_begin_capture and end_capture work together."""
        from trtutils.core._graph import (
            cuda_graph_destroy,
            cuda_stream_begin_capture,
            cuda_stream_end_capture,
        )

        cuda_stream_begin_capture(cuda_stream)
        graph = cuda_stream_end_capture(cuda_stream)
        assert graph is not None
        cuda_graph_destroy(graph)

    def test_begin_capture_with_explicit_mode(self, cuda_stream) -> None:
        """cuda_stream_begin_capture with explicit mode argument."""
        from trtutils.compat._libs import cudart
        from trtutils.core._graph import (
            cuda_graph_destroy,
            cuda_stream_begin_capture,
            cuda_stream_end_capture,
        )

        mode = cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal
        cuda_stream_begin_capture(cuda_stream, mode=mode)
        graph = cuda_stream_end_capture(cuda_stream)
        assert graph is not None
        cuda_graph_destroy(graph)

    def test_instantiate_graph(self, cuda_stream) -> None:
        """cuda_graph_instantiate creates an executable from a graph."""
        from trtutils.core._graph import (
            cuda_graph_destroy,
            cuda_graph_exec_destroy,
            cuda_graph_instantiate,
            cuda_stream_begin_capture,
            cuda_stream_end_capture,
        )

        cuda_stream_begin_capture(cuda_stream)
        graph = cuda_stream_end_capture(cuda_stream)
        graph_exec = cuda_graph_instantiate(graph)
        assert graph_exec is not None
        cuda_graph_exec_destroy(graph_exec)
        cuda_graph_destroy(graph)

    def test_graph_launch(self, cuda_stream) -> None:
        """cuda_graph_launch executes a graph without error."""
        from trtutils.core._graph import (
            cuda_graph_destroy,
            cuda_graph_exec_destroy,
            cuda_graph_instantiate,
            cuda_graph_launch,
            cuda_stream_begin_capture,
            cuda_stream_end_capture,
        )
        from trtutils.core._stream import stream_synchronize

        cuda_stream_begin_capture(cuda_stream)
        graph = cuda_stream_end_capture(cuda_stream)
        graph_exec = cuda_graph_instantiate(graph)
        cuda_graph_launch(graph_exec, cuda_stream)
        stream_synchronize(cuda_stream)
        cuda_graph_exec_destroy(graph_exec)
        cuda_graph_destroy(graph)

    def test_graph_destroy_cleanup(self, cuda_stream) -> None:
        """cuda_graph_destroy and cuda_graph_exec_destroy do not error."""
        from trtutils.core._graph import (
            cuda_graph_destroy,
            cuda_graph_exec_destroy,
            cuda_graph_instantiate,
            cuda_stream_begin_capture,
            cuda_stream_end_capture,
        )

        cuda_stream_begin_capture(cuda_stream)
        graph = cuda_stream_end_capture(cuda_stream)
        graph_exec = cuda_graph_instantiate(graph)
        cuda_graph_exec_destroy(graph_exec)  # Should not raise
        cuda_graph_destroy(graph)  # Should not raise


# ---------------------------------------------------------------------------
# CUDAGraph capture failure tests (mocked -- no GPU required)
# ---------------------------------------------------------------------------
class TestCUDAGraphCaptureFailure:
    """Tests for CUDAGraph.stop() failure paths (lines 194-207)."""

    def _make_graph(self) -> tuple:
        """Create a CUDAGraph with a mock stream."""
        from trtutils.core._graph import CUDAGraph

        mock_stream = MagicMock()
        graph = CUDAGraph(mock_stream)
        return graph, mock_stream

    @patch("trtutils.core._graph.cuda_stream_end_capture")
    @patch("trtutils.core._graph.cuda_stream_begin_capture")
    def test_stop_stream_capture_error_returns_false(
        self,
        mock_begin,
        mock_end,
    ) -> None:
        """stop() returns False when cuda_stream_end_capture raises StreamCapture error."""
        mock_end.side_effect = RuntimeError("cudaErrorStreamCapture: operation failed")
        graph, _ = self._make_graph()
        graph.start()
        result = graph.stop()
        assert result is False
        assert graph.is_captured is False

    @patch("trtutils.core._graph.cuda_stream_end_capture")
    @patch("trtutils.core._graph.cuda_stream_begin_capture")
    def test_stop_stream_capture_variant_returns_false(
        self,
        mock_begin,
        mock_end,
    ) -> None:
        """stop() returns False for 'StreamCapture' variant in error message."""
        mock_end.side_effect = RuntimeError("StreamCapture error in driver")
        graph, _ = self._make_graph()
        graph.start()
        result = graph.stop()
        assert result is False

    @patch("trtutils.core._graph.cuda_stream_end_capture")
    @patch("trtutils.core._graph.cuda_stream_begin_capture")
    def test_stop_generic_runtime_error_returns_false(
        self,
        mock_begin,
        mock_end,
    ) -> None:
        """stop() returns False for a generic RuntimeError (non-StreamCapture)."""
        mock_end.side_effect = RuntimeError("some other CUDA error")
        graph, _ = self._make_graph()
        graph.start()
        result = graph.stop()
        assert result is False
        assert graph.is_captured is False

    @patch("trtutils.core._graph.cuda_graph_destroy")
    @patch("trtutils.core._graph.cuda_graph_exec_destroy")
    @patch("trtutils.core._graph.cuda_stream_end_capture")
    @patch("trtutils.core._graph.cuda_stream_begin_capture")
    def test_stop_failure_calls_invalidate(
        self,
        mock_begin,
        mock_end,
        mock_exec_destroy,
        mock_graph_destroy,
    ) -> None:
        """stop() calls invalidate() on capture failure, clearing any partial state."""
        mock_end.side_effect = RuntimeError("capture failed")
        graph, _ = self._make_graph()
        # Simulate partial state that invalidate should clear
        graph._graph = MagicMock()
        graph._graph_exec = MagicMock()
        graph.start()
        graph.stop()
        assert graph._graph is None
        assert graph._graph_exec is None

    @patch("trtutils.core._graph.cuda_graph_destroy")
    @patch("trtutils.core._graph.cuda_graph_exec_destroy")
    @patch("trtutils.core._graph.cuda_graph_instantiate")
    @patch("trtutils.core._graph.cuda_stream_end_capture")
    @patch("trtutils.core._graph.cuda_stream_begin_capture")
    def test_stop_failure_during_instantiate(
        self,
        mock_begin,
        mock_end,
        mock_instantiate,
        mock_exec_destroy,
        mock_graph_destroy,
    ) -> None:
        """stop() handles RuntimeError from cuda_graph_instantiate."""
        mock_end.return_value = MagicMock()  # end_capture succeeds
        mock_instantiate.side_effect = RuntimeError("instantiation failed")
        graph, _ = self._make_graph()
        graph.start()
        result = graph.stop()
        assert result is False
        assert graph.is_captured is False

    @patch("trtutils.core._graph.LOG")
    @patch("trtutils.core._graph.cuda_stream_end_capture")
    @patch("trtutils.core._graph.cuda_stream_begin_capture")
    def test_stop_stream_capture_error_logs_specific_warning(
        self,
        mock_begin,
        mock_end,
        mock_log,
    ) -> None:
        """StreamCapture errors produce a specific warning about graph support."""
        mock_end.side_effect = RuntimeError("cudaErrorStreamCapture: not supported")
        graph, _ = self._make_graph()
        graph.start()
        graph.stop()
        mock_log.warning.assert_called_once()
        msg = mock_log.warning.call_args[0][0]
        assert "engine may not support graphs" in msg

    @patch("trtutils.core._graph.LOG")
    @patch("trtutils.core._graph.cuda_stream_end_capture")
    @patch("trtutils.core._graph.cuda_stream_begin_capture")
    def test_stop_generic_error_logs_generic_warning(
        self,
        mock_begin,
        mock_end,
        mock_log,
    ) -> None:
        """Non-StreamCapture errors produce a generic capture failure warning."""
        mock_end.side_effect = RuntimeError("unknown error")
        graph, _ = self._make_graph()
        graph.start()
        graph.stop()
        mock_log.warning.assert_called_once()
        msg = mock_log.warning.call_args[0][0]
        assert "CUDA graph capture failed:" in msg
        assert "engine may not support graphs" not in msg
