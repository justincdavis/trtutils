# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from trtutils.download._tools import run_cmd, run_uv_pip_install

if TYPE_CHECKING:
    from pathlib import Path


def export_torchvision_classifier(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,
    model: str,
    opset: int,
    imgsz: int | None = None,
    *,
    no_cache: bool | None = None,  # noqa: ARG001
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,  # noqa: ARG001
    verbose: bool | None = None,
) -> Path:
    _224 = 224
    if imgsz is None:
        imgsz = _224
    run_uv_pip_install(
        directory,
        bin_path.parent,
        "torchvision_classifier",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    script_content = f"""\
import torch
import torchvision.models as models
import onnx
import onnxslim

model_name = "{config["name"]}"
opset = {opset}
imgsz = {imgsz}
output_path = model_name + ".onnx"

model = getattr(models, model_name)(weights="DEFAULT")
model.eval()
dummy_input = torch.randn(1, 3, imgsz, imgsz)
torch.onnx.export(
    model,
    dummy_input,
    output_path,
    opset_version=opset,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={{"input": {{0: "batch_size"}}, "output": {{0: "batch_size"}}}},
)
# Fix Einsum equations: TensorRT only supports lowercase letters
onnx_model = onnx.load(output_path)
for node in onnx_model.graph.node:
    if node.op_type == "Einsum":
        for attr in node.attribute:
            if attr.name == "equation":
                attr.s = attr.s.lower()
onnx.save(onnx_model, output_path)

slim_model = onnxslim.slim(output_path)
onnx.save(slim_model, output_path)
"""
    script_path = directory / "_export_torchvision_classifier.py"
    script_path.write_text(script_content)
    run_cmd(
        [python_path, str(script_path)],
        cwd=directory,
        verbose=verbose,
    )
    model_path = directory / (config["name"] + ".onnx")
    new_model_path = model_path.with_name(model + model_path.suffix)
    shutil.move(model_path, new_model_path)
    return new_model_path
