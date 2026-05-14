# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
File showcasing Jetson-specific benchmarking and profiling.

Demonstrates :func:`trtutils.jetson.benchmark_engine` (latency + power + energy)
and :func:`trtutils.jetson.profile_engine` (per-layer energy contributions).
These mirror the standard :mod:`trtutils` benchmark / profile functions but
also sample VDD_TOTAL via tegrastats.

Exits cleanly when not running on a Jetson device. ``cuda_graph=False`` is
required for engines with DLA layers because DLA does not support CUDA graphs.
"""

from __future__ import annotations

from pathlib import Path

import tensorrt as trt

from trtutils import FLAGS, build_engine, set_log_level
from trtutils.download import download

# trtutils.jetson is only importable on Jetson; do the import lazily.
if FLAGS.IS_JETSON:
    from trtutils import jetson


def main() -> None:
    if not FLAGS.IS_JETSON:
        print("Skipping: not running on a Jetson device.")
        return

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

    print("Benchmarking on Jetson (measures latency + power + energy)...")
    result = jetson.benchmark_engine(
        engine_path,
        iterations=500,
        warmup_iterations=20,
        cuda_graph=False,
    )
    print(f"  latency:   mean={result.latency.mean:.3f} ms")
    print(f"  power:     mean={result.power_draw.mean:.1f} mW")
    print(f"  energy:    mean={result.energy.mean:.3f} mJ/iter")

    print("\nProfiling per-layer energy (this takes a while)...")
    prof = jetson.profile_engine(
        engine_path,
        iterations=2000,
        warmup_iterations=20,
        cuda_graph=False,
    )
    print(f"  layers profiled: {len(prof.layers)}")
    print(f"  total power:     {prof.power_draw.mean:.1f} mW")
    print(f"  total energy:    {prof.energy.mean:.3f} mJ/iter")

    hottest = sorted(prof.layers, key=lambda layer: layer.energy, reverse=True)[:5]
    print("\nTop 5 layers by energy:")
    for layer in hottest:
        print(
            f"  {layer.energy:7.3f} mJ  ({layer.power:6.1f} mW for {layer.mean:5.3f} ms)  "
            f"{layer.name}"
        )


if __name__ == "__main__":
    set_log_level("ERROR")
    main()
