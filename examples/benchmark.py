# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
File showcasing how to benchmark TensorRT engines.

Demonstrates :func:`trtutils.benchmark_engine` for a single engine and
:func:`trtutils.benchmark_engines` for side-by-side comparison. Builds an
FP32 and FP16 variant of YOLOv8n and reports latency statistics for each.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from trtutils import benchmark_engine, benchmark_engines, build_engine, set_log_level
from trtutils.download import download


def main() -> None:
    tmp_dir = Path(tempfile.gettempdir())
    onnx_path = tmp_dir / "yolov8n.onnx"
    fp32_engine = tmp_dir / "yolov8n_fp32.engine"
    fp16_engine = tmp_dir / "yolov8n_fp16.engine"

    if not onnx_path.exists():
        print("Downloading yolov8n ONNX model...")
        download("yolov8n", onnx_path, imgsz=640, simplify=True)

    shapes = [("images", (1, 3, 640, 640))]
    if not fp32_engine.exists():
        print("Building FP32 engine...")
        build_engine(onnx_path, fp32_engine, shapes=shapes)
    if not fp16_engine.exists():
        print("Building FP16 engine...")
        build_engine(onnx_path, fp16_engine, fp16=True, shapes=shapes)

    # Metric latency is in seconds -- scale to ms for readability, same as the CLI
    print("\nSingle-engine benchmarks:")
    for label, path in [("FP32", fp32_engine), ("FP16", fp16_engine)]:
        result = benchmark_engine(path, iterations=200, warmup_iterations=20)
        m = result.latency
        print(
            f"  {label}: mean={m.mean * 1000:.3f} ms  median={m.median * 1000:.3f} ms  "
            f"min={m.min * 1000:.3f} ms  max={m.max * 1000:.3f} ms"
        )

    print("\nbenchmark_engines (serial):")
    serial = benchmark_engines([fp32_engine, fp16_engine], iterations=200, warmup_iterations=20)
    for label, result in zip(["FP32", "FP16"], serial):
        print(f"  {label}: mean={result.latency.mean * 1000:.3f} ms")

    print("\nbenchmark_engines (parallel, both engines run in lockstep):")
    parallel = benchmark_engines(
        [fp32_engine, fp16_engine],
        iterations=200,
        warmup_iterations=20,
        parallel=True,
    )
    print(f"  combined: mean={parallel[0].latency.mean * 1000:.3f} ms")


if __name__ == "__main__":
    set_log_level("ERROR")
    main()
