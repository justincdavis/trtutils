# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import nvtx

from trtutils._flags import FLAGS
from trtutils._log import LOG
from trtutils.compat._libs import cudart

if TYPE_CHECKING:
    from types import TracebackType

from ._cuda import cuda_call

if TYPE_CHECKING:
    from typing_extensions import Self


def cuda_stream_begin_capture(
    stream: cudart.cudaStream_t,
    mode: cudart.cudaStreamCaptureMode | None = None,
) -> None:
    """
    Begin capturing a CUDA graph on the given stream.

    Parameters
    ----------
    stream : cudart.cudaStream_t
        The CUDA stream to begin capture on.
    mode : cudart.cudaStreamCaptureMode, optional
        The capture mode to use. Default is ThreadLocal, which only checks
        CUDA calls from the capturing thread. Global mode would cause any
        uncapturable call in any thread to fail during capture.

    """
    if mode is None:
        mode = cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal
    cuda_call(cudart.cudaStreamBeginCapture(stream, mode))


def cuda_stream_end_capture(stream: cudart.cudaStream_t) -> cudart.cudaGraph_t:
    """
    End capturing a CUDA graph and return the captured graph.

    Parameters
    ----------
    stream : cudart.cudaStream_t
        The CUDA stream to end capture on.

    Returns
    -------
    cudart.cudaGraph_t
        The captured CUDA graph.

    """
    return cuda_call(cudart.cudaStreamEndCapture(stream))


def cuda_graph_instantiate(
    graph: cudart.cudaGraph_t,
    flags: int = 0,
) -> cudart.cudaGraphExec_t:
    """
    Instantiate a CUDA graph executable.

    Parameters
    ----------
    graph : cudart.cudaGraph_t
        The CUDA graph to instantiate.
    flags : int, optional
        Flags for graph instantiation. Default is 0.

    Returns
    -------
    cudart.cudaGraphExec_t
        The instantiated graph executable.

    """
    return cuda_call(cudart.cudaGraphInstantiate(graph, flags))


def cuda_graph_launch(
    graph_exec: cudart.cudaGraphExec_t,
    stream: cudart.cudaStream_t,
) -> None:
    """
    Launch a CUDA graph executable.

    Parameters
    ----------
    graph_exec : cudart.cudaGraphExec_t
        The graph executable to launch.
    stream : cudart.cudaStream_t
        The CUDA stream to launch on.

    """
    cuda_call(cudart.cudaGraphLaunch(graph_exec, stream))


def cuda_graph_destroy(graph: cudart.cudaGraph_t) -> None:
    """
    Destroy a CUDA graph.

    Parameters
    ----------
    graph : cudart.cudaGraph_t
        The CUDA graph to destroy.

    """
    cuda_call(cudart.cudaGraphDestroy(graph))


def cuda_graph_exec_destroy(graph_exec: cudart.cudaGraphExec_t) -> None:
    """
    Destroy a CUDA graph executable.

    Parameters
    ----------
    graph_exec : cudart.cudaGraphExec_t
        The graph executable to destroy.

    """
    cuda_call(cudart.cudaGraphExecDestroy(graph_exec))


class CUDAGraph:
    """Wrapper around CUDA graph capture and execution."""

    def __init__(self: Self, stream: cudart.cudaStream_t) -> None:
        """
        Initialize the CUDA graph helper.

        Parameters
        ----------
        stream : cudart.cudaStream_t
            The CUDA stream to use for graph operations.

        """
        self._stream = stream
        self._graph: cudart.cudaGraph_t | None = None
        self._graph_exec: cudart.cudaGraphExec_t | None = None

    def __enter__(self: Self) -> Self:
        self.start()
        return self

    def __exit__(
        self: Self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        success = self.stop()
        # __exit__ returns False on success, kind of funky
        return not success

    def __del__(self: Self) -> None:
        self.invalidate()

    def start(self: Self) -> None:
        """
        Begin graph capture.

        This should be called before the operations to capture.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range("cuda_graph::start")
        cuda_stream_begin_capture(self._stream)
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()

    def stop(self: Self) -> bool:
        """
        End graph capture and instantiate the graph.

        Returns
        -------
        bool
            True if capture and instantiation succeeded, False otherwise.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range("cuda_graph::stop")
        try:
            self._graph = cuda_stream_end_capture(self._stream)
            self._graph_exec = cuda_graph_instantiate(self._graph, 0)
        except RuntimeError as e:
            err_str = str(e)
            if "cudaErrorStreamCapture" in err_str or "StreamCapture" in err_str:
                LOG.warning(
                    f"CUDA graph capture failed (engine may not support graphs): {err_str}",
                )
            else:
                LOG.warning(f"CUDA graph capture failed: {err_str}")

            self.invalidate()

            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()
            return False
        else:
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()
            return True

    def launch(self: Self) -> None:
        """
        Launch the captured graph.

        Raises
        ------
        RuntimeError
            If no graph has been captured.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range("cuda_graph::launch")
        if self._graph_exec is None:
            err_msg = "Cannot launch graph: no graph has been captured"
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()
            raise RuntimeError(err_msg)
        cuda_graph_launch(self._graph_exec, self._stream)
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()

    def invalidate(self: Self) -> None:
        """Destroy the graph and graph executable, resetting state."""
        with contextlib.suppress(AttributeError, RuntimeError):
            cuda_graph_exec_destroy(self._graph_exec)
            self._graph_exec = None
        with contextlib.suppress(AttributeError, RuntimeError):
            cuda_graph_destroy(self._graph)
            self._graph = None

    @property
    def is_captured(self: Self) -> bool:
        """
        Check if a graph has been captured.

        Returns
        -------
        bool
            True if a graph has been captured, False otherwise.

        """
        return self._graph_exec is not None
