# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
File showcasing a high-level tour of the ``trtutils.core`` CUDA backend.

Demonstrates :class:`trtutils.core.Device`, device introspection
(:func:`trtutils.core.get_device`, :func:`trtutils.core.get_device_name`,
:func:`trtutils.core.get_compute_capability`), stream lifecycle
(:func:`trtutils.core.create_stream` / :func:`trtutils.core.destroy_stream`,
:func:`trtutils.core.stream_synchronize`), explicit device memory
(:func:`trtutils.core.cuda_malloc`, :func:`trtutils.core.cuda_free`,
:func:`trtutils.core.memcpy_host_to_device` /
:func:`trtutils.core.memcpy_device_to_host`), and :class:`trtutils.core.CUDAGraph`
capture/replay wrapped around a synthetic stream sleep.

The goal is to show that the ``core`` module exists and to give a guided
overview of its building blocks; it is not a deep dive into any one piece.
"""

from __future__ import annotations

import numpy as np

from trtutils import set_log_level
from trtutils.core import (
    CUDAGraph,
    Device,
    create_stream,
    cuda_free,
    cuda_malloc,
    destroy_stream,
    get_compute_capability,
    get_device,
    get_device_count,
    get_device_name,
    get_num_dla_cores,
    memcpy_device_to_host,
    memcpy_host_to_device,
    memcpy_host_to_device_async,
    stream_synchronize,
)


def main() -> None:
    print("Device info:")
    print(f"  current device index: {get_device()}")
    print(f"  device count:         {get_device_count()}")
    print(f"  device name:          {get_device_name()}")
    print(f"  compute capability:   {get_compute_capability()}")
    print(f"  DLA cores:            {get_num_dla_cores()}")

    # Device(idx) saves/restores the current device on exit; Device(None) is a no-op.
    with Device(get_device()):
        print(f"  inside Device guard:  {get_device()}")

    # synchronous memcpy roundtrip
    host = np.arange(8, dtype=np.float32)
    nbytes = host.nbytes
    device_ptr = cuda_malloc(nbytes)
    memcpy_host_to_device(device_ptr, host)
    roundtrip = np.zeros_like(host)
    memcpy_device_to_host(roundtrip, device_ptr)
    cuda_free(device_ptr)
    print(f"\nSync memcpy roundtrip: {host.tolist()} -> {roundtrip.tolist()}")

    # async memcpy through a stream
    stream = create_stream()
    device_ptr = cuda_malloc(nbytes)
    memcpy_host_to_device_async(device_ptr, host, stream)
    stream_synchronize(stream)
    memcpy_device_to_host(roundtrip, device_ptr)
    print(f"Async memcpy result:   {roundtrip.tolist()}")
    cuda_free(device_ptr)

    # CUDA graph capture/replay around an async memcpy
    src = np.arange(16, dtype=np.float32)
    dst = np.zeros_like(src)
    device_ptr = cuda_malloc(src.nbytes)
    graph = CUDAGraph(stream)
    with graph:
        memcpy_host_to_device_async(device_ptr, src, stream)
    if graph.is_captured:
        graph.launch()
        stream_synchronize(stream)
        memcpy_device_to_host(dst, device_ptr)
        print(f"\nCUDA graph captured and replayed: dst[:4]={dst[:4].tolist()}")
    else:
        print("\nCUDA graph capture failed on this stream (skipping launch).")
    graph.invalidate()
    cuda_free(device_ptr)
    destroy_stream(stream)


if __name__ == "__main__":
    set_log_level("ERROR")
    main()
