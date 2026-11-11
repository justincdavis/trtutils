# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/core/_graph.py -- CUDA graph capture and replay."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from trtutils.compat._libs import cudart
from trtutils.core import _graph
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


def test_cuda_graph_invalidate_destroys_each_handle_once(cuda_stream, monkeypatch) -> None:
    """Repeated invalidate() and __del__ never destroy a null or already-destroyed handle."""
    destroyed = []
    monkeypatch.setattr(_graph, "cuda_graph_destroy", destroyed.append)
    monkeypatch.setattr(_graph, "cuda_graph_exec_destroy", destroyed.append)
    graph = CUDAGraph(cuda_stream)
    graph.invalidate()
    assert destroyed == []
    graph._graph, graph._graph_exec = "graph", "exec"
    graph.invalidate()
    graph.invalidate()
    graph.__del__()
    assert destroyed == ["exec", "graph"]


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


def test_cuda_graph_invalidate_destroys_graph_when_exec_destroy_fails(
    cuda_stream, monkeypatch
) -> None:
    """A failing exec destroy still destroys the graph."""
    graph = CUDAGraph(cuda_stream)
    with graph:
        pass
    destroyed = []
    real_destroy = _graph.cuda_graph_destroy

    def fail_exec(_exec) -> None:
        msg = "boom"
        raise RuntimeError(msg)

    def spy(g) -> None:
        destroyed.append(g)
        real_destroy(g)

    monkeypatch.setattr(_graph, "cuda_graph_exec_destroy", fail_exec)
    monkeypatch.setattr(_graph, "cuda_graph_destroy", spy)
    graph.invalidate()
    assert len(destroyed) == 1


@pytest.mark.parametrize("nvtx_enabled", [True, False])
def test_cuda_graph_start_failure_balances_nvtx(cuda_stream, monkeypatch, nvtx_enabled) -> None:
    """A failed begin-capture does not leave the NVTX range open."""
    ranges: list[str | None] = []
    monkeypatch.setattr(_graph.FLAGS, "NVTX_ENABLED", nvtx_enabled)
    monkeypatch.setattr(
        _graph,
        "nvtx",
        SimpleNamespace(push_range=ranges.append, pop_range=lambda: ranges.append(None)),
    )

    def fail(_stream) -> None:
        msg = "boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(_graph, "cuda_stream_begin_capture", fail)
    with pytest.raises(RuntimeError, match="boom"):
        CUDAGraph(cuda_stream).start()
    assert ranges.count(None) == len(ranges) - ranges.count(None)
