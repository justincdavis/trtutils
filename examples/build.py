# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
File showcasing how to build a TensorRT engine from an ONNX model.

Demonstrates :func:`trtutils.builder.read_onnx` for peeking at the network
before building, then :func:`trtutils.build_engine` to produce a serialized
engine file. Bootstraps its own ONNX via :func:`trtutils.download.download`.
"""

from __future__ import annotations

import time
from pathlib import Path

from trtutils import build_engine, set_log_level
from trtutils.builder import read_onnx
from trtutils.download import download


def main() -> None:
    onnx_path = Path("/tmp/yolov8n.onnx")  # noqa: S108
    engine_path = Path("/tmp/yolov8n.engine")  # noqa: S108

    if not onnx_path.exists():
        print("Downloading yolov8n ONNX model...")
        download("yolov8n", onnx_path, imgsz=640, simplify=True)

    # peek at the parsed network before we build
    network, _builder, _config, _parser = read_onnx(onnx_path)
    print(f"ONNX network: {network.num_layers} layers")
    for i in range(network.num_inputs):
        t = network.get_input(i)
        print(f"  input {t.name}: shape={tuple(t.shape)}, dtype={t.dtype}")
    for i in range(network.num_outputs):
        t = network.get_output(i)
        print(f"  output {t.name}: shape={tuple(t.shape)}, dtype={t.dtype}")
    # release the parser-side handles before building
    del network, _builder, _config, _parser

    if engine_path.exists():
        engine_path.unlink()

    t0 = time.perf_counter()
    build_engine(
        onnx_path,
        engine_path,
        fp16=True,
        shapes=[("images", (1, 3, 640, 640))],
    )
    t1 = time.perf_counter()

    size_mb = engine_path.stat().st_size / (1024 * 1024)
    print(f"Built FP16 engine in {t1 - t0:.2f} s -> {engine_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    set_log_level("ERROR")
    main()
