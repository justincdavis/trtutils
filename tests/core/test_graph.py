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


def _make_mock_graph() -> tuple:
    """Create a CUDAGraph with a mock stream."""
    mock_stream = MagicMock()
    return CUDAGraph(mock_stream), mock_stream


def test_cuda_graph_not_captured(cuda_stream) -> None:
    """Fresh CUDAGraph is not captured."""
    graph = CUDAGraph(cuda_stream)
    assert graph.is_captured is False


def test_cuda_graph_capture_and_launch(cuda_stream) -> None:
    """Context manager capture, manual start/stop, launch, and invalidate lifecycle."""
    graph = CUDAGraph(cuda_stream)
    # context manager marks captured
    with graph:
        pass
    assert graph.is_captured is True
    graph.invalidate()
    assert graph.is_captured is False
    # manual start/stop marks captured
    graph.start()
    success = graph.stop()
    assert success is True
    assert graph.is_captured is True
    # launch after capture works
    graph.launch()
    stream_synchronize(cuda_stream)
    graph.invalidate()
    # second capture cycle works
    with graph:
        pass
    assert graph.is_captured is True
    graph.invalidate()


def test_cuda_graph_launch_without_capture_raises(cuda_stream) -> None:
    """launch() without capture raises RuntimeError."""
    graph = CUDAGraph(cuda_stream)
    with pytest.raises(RuntimeError, match="no graph has been captured"):
        graph.launch()


def test_cuda_graph_invalidate_idempotent(cuda_stream) -> None:
    """invalidate() on fresh or already-invalidated graph does not raise."""
    graph = CUDAGraph(cuda_stream)
    graph.invalidate()
    with graph:
        pass
    graph.invalidate()
    graph.invalidate()


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param(None, id="default_mode"),
        pytest.param("thread_local", id="explicit_mode"),
    ],
)
def test_cuda_graph_low_level_lifecycle(cuda_stream, mode) -> None:
    """begin_capture, end_capture, instantiate, launch, destroy all work together."""
    if mode == "thread_local":
        mode_val = cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal
        cuda_stream_begin_capture(cuda_stream, mode=mode_val)
    else:
        cuda_stream_begin_capture(cuda_stream)
    graph = cuda_stream_end_capture(cuda_stream)
    assert graph is not None
    # instantiate and launch
    graph_exec = cuda_graph_instantiate(graph)
    assert graph_exec is not None
    cuda_graph_launch(graph_exec, cuda_stream)
    stream_synchronize(cuda_stream)
    # cleanup
    cuda_graph_exec_destroy(graph_exec)
    cuda_graph_destroy(graph)


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
def test_cuda_graph_stop_returns_false_on_error(mock_begin, mock_end, error_msg) -> None:
    """stop() returns False when cuda_stream_end_capture raises."""
    mock_end.side_effect = RuntimeError(error_msg)
    graph, _ = _make_mock_graph()
    graph.start()
    result = graph.stop()
    assert result is False
    assert graph.is_captured is False


@patch("trtutils.core._graph.cuda_graph_destroy")
@patch("trtutils.core._graph.cuda_graph_exec_destroy")
@patch("trtutils.core._graph.cuda_stream_end_capture")
@patch("trtutils.core._graph.cuda_stream_begin_capture")
def test_cuda_graph_stop_failure_calls_invalidate(
    mock_begin, mock_end, mock_exec_destroy, mock_graph_destroy
) -> None:
    """stop() clears partial state on capture failure."""
    mock_end.side_effect = RuntimeError("capture failed")
    graph, _ = _make_mock_graph()
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
def test_cuda_graph_stop_failure_during_instantiate(
    mock_begin, mock_end, mock_instantiate, mock_exec_destroy, mock_graph_destroy
) -> None:
    """stop() handles RuntimeError from cuda_graph_instantiate."""
    mock_end.return_value = MagicMock()
    mock_instantiate.side_effect = RuntimeError("instantiation failed")
    graph, _ = _make_mock_graph()
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
def test_cuda_graph_stop_logs_warning(
    mock_begin, mock_end, mock_log, error_msg, expected_fragment, excluded_fragment
) -> None:
    """stop() logs appropriate warning based on error type."""
    mock_end.side_effect = RuntimeError(error_msg)
    graph, _ = _make_mock_graph()
    graph.start()
    graph.stop()
    mock_log.warning.assert_called_once()
    msg = mock_log.warning.call_args[0][0]
    assert expected_fragment in msg
    if excluded_fragment is not None:
        assert excluded_fragment not in msg
