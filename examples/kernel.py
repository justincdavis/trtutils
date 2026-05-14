# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
File showcasing the :class:`trtutils.core.Kernel` abstraction.

Compiles a small CUDA kernel via NVRTC, allocates a device buffer, runs the
kernel on a stream, and copies the result back. ``Kernel`` reads CUDA source
from a ``.cu`` file and exposes ``create_args`` / ``__call__`` for ergonomic
launches.

Self-contained: writes the kernel source to a temp file, no engine required.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from trtutils import set_log_level
from trtutils.core import (
    Kernel,
    create_stream,
    cuda_free,
    cuda_malloc,
    destroy_stream,
    memcpy_device_to_host,
    stream_synchronize,
)

ADD_ONE_KERNEL = """\
extern "C" __global__ void add_one(float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (float)idx + 1.0f;
    }
}
"""


def main() -> None:
    # Write the CUDA source to a temp file — Kernel takes a file path.
    with tempfile.NamedTemporaryFile("w", suffix=".cu", delete=False) as f:
        f.write(ADD_ONE_KERNEL)
        cu_path = Path(f.name)

    kernel_name = "add_one"
    kernel = Kernel(cu_path, name=kernel_name)
    print(f"Compiled kernel '{kernel_name}' from {cu_path}")

    n = 32
    out_bytes = n * np.dtype(np.float32).itemsize
    d_out = cuda_malloc(out_bytes)
    stream = create_stream()

    # create_args boxes Python ints/floats and device pointers into a uint64
    # pointer array the CUDA launch ABI expects.
    args = kernel.create_args(d_out, n)

    # 1 block of n threads
    kernel((1, 1, 1), (n, 1, 1), stream, args)
    stream_synchronize(stream)

    result = np.zeros(n, dtype=np.float32)
    memcpy_device_to_host(result, d_out)
    print(f"Kernel output (first 8 of {n}): {result[:8].tolist()}")
    print(
        f"Matches expected (idx + 1): {np.array_equal(result, np.arange(1, n + 1, dtype=np.float32))}"
    )

    kernel.free()
    cuda_free(d_out)
    destroy_stream(stream)
    cu_path.unlink(missing_ok=True)


if __name__ == "__main__":
    set_log_level("ERROR")
    main()
