# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
File showcasing how to inspect TensorRT engines and ONNX models.

Demonstrates :func:`trtutils.inspect_engine`, :func:`trtutils.inspect.get_engine_names`,
and :func:`trtutils.inspect.inspect_onnx_layers` — first peeking at a built
engine, then walking the source ONNX layer-by-layer to show the ONNX-to-TRT
fusion mapping.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from trtutils import build_engine, inspect_engine, set_log_level
from trtutils.download import download
from trtutils.inspect import get_engine_names, inspect_onnx_layers


def main() -> None:
    tmp_dir = Path(tempfile.gettempdir())
    onnx_path = tmp_dir / "yolov8n.onnx"
    engine_path = tmp_dir / "yolov8n.engine"

    if not onnx_path.exists():
        print("Downloading yolov8n ONNX model...")
        download("yolov8n", onnx_path, imgsz=640, simplify=True)
    if not engine_path.exists():
        print("Building yolov8n engine...")
        build_engine(onnx_path, engine_path, fp16=True, shapes=[("images", (1, 3, 640, 640))])

    # high-level engine summary
    mem_size, batch_size, inputs, outputs = inspect_engine(engine_path)
    print(f"Engine: {engine_path.name}")
    print(f"  device memory: {mem_size / (1024 * 1024):.2f} MB")
    print(f"  max batch size: {batch_size}")
    print(f"  inputs ({len(inputs)}):")
    for name, shape, dtype, fmt in inputs:
        print(f"    {name}: shape={tuple(shape)} dtype={dtype} format={fmt}")
    print(f"  outputs ({len(outputs)}):")
    for name, shape, dtype, fmt in outputs:
        print(f"    {name}: shape={tuple(shape)} dtype={dtype} format={fmt}")

    # input/output names in enumeration order
    in_names, out_names = get_engine_names(engine_path)
    print(f"Names: inputs={in_names}, outputs={out_names}")

    # walk the ONNX layers and show DLA compatibility per layer
    onnx_layers = inspect_onnx_layers(onnx_path)
    print(f"\nONNX layers: {len(onnx_layers)} total")
    dla_count = sum(1 for layer in onnx_layers if layer.dla_compatible)
    print(f"  DLA-compatible: {dla_count}/{len(onnx_layers)}")
    print("First 5 layers:")
    for layer in onnx_layers[:5]:
        tag = "DLA" if layer.dla_compatible else "GPU"
        print(
            f"  [{layer.index:3d}] {layer.layer_type:<16} {layer.name}  "
            f"out={layer.output_tensor_size}B  {tag}"
        )


if __name__ == "__main__":
    set_log_level("ERROR")
    main()
