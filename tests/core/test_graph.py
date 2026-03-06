# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/core/_graph.py -- CUDA graph capture and replay."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from trtutils.compat._libs import cudart
from trtutils.core._graph import (
    CUDAGraph,
    cuda_graph_destroy,
    cuda_graph_exec_destroy,
    cuda_graph_instantiate,
    cuda_graph_launch,
    cuda_stream_begin_capture,
    cuda_stream_end_capture,
)
from trtutils.core._stream import stream_synchronize


# ---------------------------------------------------------------------------
# CUDAGraph class lifecycle tests
# ---------------------------------------------------------------------------
class TestCUDAGraphLifecycle:
    """Tests for CUDAGraph class init, capture, launch, and invalidate."""

    def test_init_not_captured(self, cuda_stream) -> None:
        """After init, is_captured should be False."""
        graph = CUDAGraph(cuda_stream)
        assert graph.is_captured is False

    def test_context_manager_capture(self, cuda_stream) -> None:
        """Context manager start/stop marks graph as captured."""
        graph = CUDAGraph(cuda_stream)
        with graph:
            pass  # Empty capture -- captures an empty graph
        assert graph.is_captured is True
        graph.invalidate()

    def test_start_stop_capture(self, cuda_stream) -> None:
        """Manual start() and stop() marks graph as captured."""
        graph = CUDAGraph(cuda_stream)
        graph.start()
        success = graph.stop()
        assert success is True
        assert graph.is_captured is True
        graph.invalidate()

    def test_launch_after_capture(self, cuda_stream) -> None:
        """launch() succeeds after a successful capture."""
        graph = CUDAGraph(cuda_stream)
        with graph:
            pass  # Empty capture
        graph.launch()
        stream_synchronize(cuda_stream)
        graph.invalidate()

    def test_invalidate_clears_state(self, cuda_stream) -> None:
        """invalidate() sets is_captured back to False."""
        graph = CUDAGraph(cuda_stream)
        with graph:
            pass
        assert graph.is_captured is True
        graph.invalidate()
        assert graph.is_captured is False

    def test_launch_without_capture_raises(self, cuda_stream) -> None:
        """launch() without capture raises RuntimeError."""
        graph = CUDAGraph(cuda_stream)
        with pytest.raises(RuntimeError, match="no graph has been captured"):
            graph.launch()

    def test_invalidate_without_capture(self, cuda_stream) -> None:
        """invalidate() on a fresh graph does not raise."""
        graph = CUDAGraph(cuda_stream)
        graph.invalidate()  # Should not raise

    def test_invalidate_idempotent(self, cuda_stream) -> None:
        """invalidate() can be called multiple times safely."""
        graph = CUDAGraph(cuda_stream)
        with graph:
            pass
        graph.invalidate()
        graph.invalidate()  # Second call should not raise

    def test_del_calls_invalidate(self, cuda_stream) -> None:
        """__del__ cleans up graph resources without error."""
        graph = CUDAGraph(cuda_stream)
        with graph:
            pass
        assert graph.is_captured is True
        del graph  # Should not crash

    def test_context_manager_returns_self(self, cuda_stream) -> None:
        """__enter__ returns the CUDAGraph instance itself."""
        graph = CUDAGraph(cuda_stream)
        with graph as g:
            assert g is graph
        graph.invalidate()

    def test_multiple_capture_cycles(self, cuda_stream) -> None:
        """A graph can be invalidated and recaptured."""
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
class TestCUDAGraphFunctions:
    """Tests for the low-level CUDA graph functions."""

    @pytest.mark.parametrize(
        "mode",
        [
            pytest.param(None, id="default_mode"),
            pytest.param("thread_local", id="explicit_mode"),
        ],
    )
    def test_begin_end_capture(self, cuda_stream, mode) -> None:
        """cuda_stream_begin_capture and end_capture work together."""
        if mode == "thread_local":
            mode_val = cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal
            cuda_stream_begin_capture(cuda_stream, mode=mode_val)
        else:
            cuda_stream_begin_capture(cuda_stream)
        graph = cuda_stream_end_capture(cuda_stream)
        assert graph is not None
        cuda_graph_destroy(graph)

    def test_instantiate_graph(self, cuda_stream) -> None:
        """cuda_graph_instantiate creates an executable from a graph."""
        cuda_stream_begin_capture(cuda_stream)
        graph = cuda_stream_end_capture(cuda_stream)
        graph_exec = cuda_graph_instantiate(graph)
        assert graph_exec is not None
        cuda_graph_exec_destroy(graph_exec)
        cuda_graph_destroy(graph)

    def test_graph_launch(self, cuda_stream) -> None:
        """cuda_graph_launch executes a graph without error."""
        cuda_stream_begin_capture(cuda_stream)
        graph = cuda_stream_end_capture(cuda_stream)
        graph_exec = cuda_graph_instantiate(graph)
        cuda_graph_launch(graph_exec, cuda_stream)
        stream_synchronize(cuda_stream)
        cuda_graph_exec_destroy(graph_exec)
        cuda_graph_destroy(graph)

    def test_graph_destroy_cleanup(self, cuda_stream) -> None:
        """cuda_graph_destroy and cuda_graph_exec_destroy do not error."""
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
        mock_stream = MagicMock()
        graph = CUDAGraph(mock_stream)
        return graph, mock_stream

    @pytest.mark.parametrize(
        "error_msg",
        [
            pytest.param("cudaErrorStreamCapture: operation failed", id="stream_capture"),
            pytest.param("StreamCapture error in driver", id="stream_capture_variant"),
            pytest.param("some other CUDA error", id="generic"),
        ],
    )
    @patch("trtutils.core._graph.cuda_stream_end_capture")
    @patch("trtutils.core._graph.cuda_stream_begin_capture")
    def test_stop_returns_false_on_error(self, mock_begin, mock_end, error_msg) -> None:
        """stop() returns False when cuda_stream_end_capture raises."""
        mock_end.side_effect = RuntimeError(error_msg)
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

    @pytest.mark.parametrize(
        ("error_msg", "expected_fragment", "excluded_fragment"),
        [
            pytest.param(
                "cudaErrorStreamCapture: not supported",
                "engine may not support graphs",
                None,
                id="stream_capture_warning",
            ),
            pytest.param(
                "unknown error",
                "CUDA graph capture failed:",
                "engine may not support graphs",
                id="generic_warning",
            ),
        ],
    )
    @patch("trtutils.core._graph.LOG")
    @patch("trtutils.core._graph.cuda_stream_end_capture")
    @patch("trtutils.core._graph.cuda_stream_begin_capture")
    def test_stop_logs_warning(
        self, mock_begin, mock_end, mock_log, error_msg, expected_fragment, excluded_fragment
    ) -> None:
        """stop() logs appropriate warning based on error type."""
        mock_end.side_effect = RuntimeError(error_msg)
        graph, _ = self._make_graph()
        graph.start()
        graph.stop()
        mock_log.warning.assert_called_once()
        msg = mock_log.warning.call_args[0][0]
        assert expected_fragment in msg
        if excluded_fragment is not None:
            assert excluded_fragment not in msg
