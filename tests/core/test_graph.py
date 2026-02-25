"""Tests for src/trtutils/core/_graph.py -- CUDA graph capture and replay."""

from __future__ import annotations

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
