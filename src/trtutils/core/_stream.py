# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib

import nvtx

with contextlib.suppress(Exception):
    from trtutils.compat._libs import cudart

from trtutils._flags import FLAGS

from ._cuda import cuda_call


def create_stream() -> cudart.cudaStream_t:
    """
    Create a CUDA Stream.

    Returns
    -------
    cudart.cudaStream_t
        The CUDA stream.

    """
    if FLAGS.NVTX_ENABLED:
        nvtx.push_range("core::create_stream")
    result = cuda_call(cudart.cudaStreamCreate())
    if FLAGS.NVTX_ENABLED:
        nvtx.pop_range()
    return result


def destroy_stream(stream: cudart.cudaStream_t) -> None:
    """
    Destroy a CUDA Stream.

    Parameters
    ----------
    stream : cudart.cudaStream_t
        The CUDA stream to destroy.

    """
    cuda_call(cudart.cudaStreamDestroy(stream))


def stream_synchronize(stream: cudart.cudaStream_t) -> None:
    """
    Copy a numpy array to a device pointer with error checking.

    Parameters
    ----------
    stream : cudart.cudaStream_t
        The stream to synchronize calls for.

    """
    if FLAGS.NVTX_ENABLED:
        nvtx.push_range("core::stream_synchronize")
    cuda_call(cudart.cudaStreamSynchronize(stream))
    if FLAGS.NVTX_ENABLED:
        nvtx.pop_range()


def create_event() -> cudart.cudaEvent_t:
    """
    Create a CUDA event (timing disabled).

    Returns
    -------
    cudart.cudaEvent_t
        The CUDA event.

    """
    return cuda_call(
        cudart.cudaEventCreateWithFlags(cudart.cudaEventDisableTiming),
    )


def destroy_event(event: cudart.cudaEvent_t) -> None:
    """
    Destroy a CUDA event.

    Parameters
    ----------
    event : cudart.cudaEvent_t
        The CUDA event to destroy.

    """
    cuda_call(cudart.cudaEventDestroy(event))


def record_event(
    event: cudart.cudaEvent_t,
    stream: cudart.cudaStream_t,
) -> None:
    """
    Record a CUDA event on a stream.

    Parameters
    ----------
    event : cudart.cudaEvent_t
        The CUDA event to record.
    stream : cudart.cudaStream_t
        The stream to record the event on.

    """
    cuda_call(cudart.cudaEventRecord(event, stream))


def stream_wait_event(
    stream: cudart.cudaStream_t,
    event: cudart.cudaEvent_t,
) -> None:
    """
    Make a stream wait on a CUDA event.

    Parameters
    ----------
    stream : cudart.cudaStream_t
        The stream which will wait.
    event : cudart.cudaEvent_t
        The CUDA event to wait on.

    """
    cuda_call(cudart.cudaStreamWaitEvent(stream, event, 0))


def event_synchronize(event: cudart.cudaEvent_t) -> None:
    """
    Block the host until a CUDA event has completed.

    Parameters
    ----------
    event : cudart.cudaEvent_t
        The CUDA event to synchronize on.

    """
    cuda_call(cudart.cudaEventSynchronize(event))
