# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
File showcasing trtutils configuration, logging, and profiling toggles.

Demonstrates :data:`trtutils.FLAGS`, :data:`trtutils.CONFIG`,
:func:`trtutils.set_log_level`, the :data:`trtutils.NVTX` context manager,
and :func:`trtutils.register_jit` / :data:`trtutils.JIT` for Numba JIT
compilation. No model needed.
"""

from __future__ import annotations

import time

import numpy as np

from trtutils import (
    CONFIG,
    FLAGS,
    JIT,
    NVTX,
    register_jit,
    set_log_level,
)


@register_jit(fastmath=True)
def sum_squares(arr: np.ndarray) -> float:
    """Trivial numeric kernel — Numba JITs this when JIT is enabled."""
    total = 0.0
    for value in arr:
        total += value * value
    return float(total)


def main() -> None:
    print("FLAGS:")
    for attr in sorted(
        a for a in dir(FLAGS) if not a.startswith("_") and not callable(getattr(FLAGS, a))
    ):
        print(f"  {attr}: {getattr(FLAGS, attr)}")

    print("\nCONFIG: loading TensorRT plugins (idempotent)...")
    CONFIG.load_plugins()
    print("CONFIG: plugins loaded.")

    print("\nLog level demo — toggle between INFO and ERROR:")
    set_log_level("INFO")
    print("  log level set to INFO (TensorRT messages would print here)")
    set_log_level("ERROR")
    print("  log level set back to ERROR")

    print("\nNVTX context manager — ranges are visible to Nsight Systems:")
    with NVTX("example::demo"):
        time.sleep(0.001)
    print(f"  NVTX_ENABLED after context exit: {FLAGS.NVTX_ENABLED}")

    print("\nJIT context manager — toggle Numba compilation around a hot loop:")
    data = np.random.default_rng(0).standard_normal(100_000).astype(np.float32)

    # Warm up either path so we measure steady-state cost
    sum_squares(data)
    t0 = time.perf_counter()
    sum_squares(data)
    no_jit_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  baseline (JIT={FLAGS.JIT}): {no_jit_ms:.3f} ms")

    with JIT:
        sum_squares(data)  # one warmup so the JIT compile cost is excluded
        t0 = time.perf_counter()
        sum_squares(data)
        with_jit_ms = (time.perf_counter() - t0) * 1000.0
        print(
            f"  inside JIT block (JIT={FLAGS.JIT}, Numba={FLAGS.FOUND_NUMBA}): {with_jit_ms:.3f} ms"
        )


if __name__ == "__main__":
    set_log_level("ERROR")
    main()
