# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
File showcasing how to build a TensorRT engine targeting the DLA.

Demonstrates :func:`trtutils.builder.can_run_on_dla` to inspect which layers of
an ONNX model are DLA-compatible, and :func:`trtutils.builder.build_dla_engine`
to build a hybrid DLA/GPU engine. INT8 calibration is mandatory for DLA builds,
so we feed a :class:`trtutils.builder.SyntheticBatcher`.

Exits cleanly when the system has no DLA hardware.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from trtutils import FLAGS, TRTEngine, set_log_level
from trtutils.builder import SyntheticBatcher, build_dla_engine, can_run_on_dla
from trtutils.download import download


def main() -> None:
    if not FLAGS.HAS_DLA:
        print(f"Skipping: no DLA cores available (NUM_DLA_CORES={FLAGS.NUM_DLA_CORES}).")
        return

    onnx_path = Path("/tmp/yolov8n.onnx")  # noqa: S108
    engine_path = Path("/tmp/yolov8n_dla.engine")  # noqa: S108

    if not onnx_path.exists():
        print("Downloading yolov8n ONNX model...")
        download("yolov8n", onnx_path, imgsz=640, simplify=True)

    full_dla, chunks = can_run_on_dla(onnx_path)
    print(f"Fully DLA-compatible: {full_dla}")
    print(f"Found {len(chunks)} layer chunks:")
    for i, (layers, start, end, on_dla) in enumerate(chunks):
        target = "DLA" if on_dla else "GPU"
        print(f"  chunk {i}: layers [{start}-{end}] ({len(layers)} layers) -> {target}")

    # DLA builds need INT8 calibration data; use synthetic data for the demo
    batcher = SyntheticBatcher(
        shape=(640, 640, 3),
        dtype=np.float32,
        batch_size=1,
        num_batches=8,
    )

    if engine_path.exists():
        engine_path.unlink()

    t0 = time.perf_counter()
    build_dla_engine(
        onnx_path,
        engine_path,
        data_batcher=batcher,
        dla_core=0,
        shapes=[("images", (1, 3, 640, 640))],
    )
    t1 = time.perf_counter()

    size_mb = engine_path.stat().st_size / (1024 * 1024)
    print(f"Built DLA engine in {t1 - t0:.2f} s -> {engine_path} ({size_mb:.2f} MB)")

    # confirm the engine is loadable; cuda_graph=False since DLA + graphs don't mix
    engine = TRTEngine(engine_path, dla_core=0, warmup=True, cuda_graph=False)
    engine.mock_execute()
    print(f"Loaded {engine.name}, mock_execute OK")
    del engine


if __name__ == "__main__":
    set_log_level("ERROR")
    main()
