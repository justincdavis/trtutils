# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/core/_graph.py -- CUDA graph capture and replay."""

from __future__ import annotations

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
