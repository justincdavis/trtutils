# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
File showcasing how to profile a TensorRT engine layer-by-layer.

Demonstrates :func:`trtutils.profile_engine` for per-layer timing statistics
and :func:`trtutils.profiling.identify_quantize_speedups_by_layer` for
detecting layers that benefit from INT8 quantization.

For useful per-layer names, the engine must be built with
``profiling_verbosity=trt.ProfilingVerbosity.DETAILED``.
"""

from __future__ import annotations

from pathlib import Path

import tensorrt as trt

from trtutils import build_engine, profile_engine, set_log_level
from trtutils.download import download
from trtutils.profiling import identify_quantize_speedups_by_layer


def main() -> None:
    onnx_path = Path("/tmp/yolov8n.onnx")  # noqa: S108
    engine_path = Path("/tmp/yolov8n_detailed.engine")  # noqa: S108

    if not onnx_path.exists():
        print("Downloading yolov8n ONNX model...")
        download("yolov8n", onnx_path, imgsz=640, simplify=True)

    if not engine_path.exists():
        print("Building yolov8n engine with DETAILED profiling verbosity...")
        build_engine(
            onnx_path,
            engine_path,
            fp16=True,
            shapes=[("images", (1, 3, 640, 640))],
            profiling_verbosity=trt.ProfilingVerbosity.DETAILED,
        )

    result = profile_engine(engine_path, iterations=100, warmup_iterations=10)
    print(f"Profiled {result.iterations} iterations across {len(result.layers)} layers")
    print(f"Total per-iteration time: mean={result.total_time.mean:.3f} ms")

    top_n = 10
    slowest = sorted(result.layers, key=lambda layer: layer.mean, reverse=True)[:top_n]
    print(f"\nTop {top_n} slowest layers:")
    for layer in slowest:
        print(f"  {layer.mean:7.3f} ms  {layer.name}")

    print("\nScanning for INT8 quantization speedups (this builds both FP16 + INT8 engines)...")
    _fp16, _int8, speedups = identify_quantize_speedups_by_layer(
        onnx_path,
        iterations=50,
        warmup_iterations=5,
    )
    quantize_wins = sorted(speedups, key=lambda pair: pair[1], reverse=True)[:5]
    print("Top 5 INT8 wins (positive % means INT8 faster):")
    for name, speedup in quantize_wins:
        print(f"  {speedup:+6.2f}%  {name}")


if __name__ == "__main__":
    set_log_level("ERROR")
    main()
