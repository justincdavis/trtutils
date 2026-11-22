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
    batch: int = 1,
    dynamic: bool = False,
    no_cache: bool | None = None,  # noqa: ARG001
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,  # noqa: ARG001
    verbose: bool | None = None,
) -> Path:
    _224 = 224
    if imgsz is None:
        imgsz = _224
    dynamic_axes = '{"input": {0: "batch_size"}, "output": {0: "batch_size"}}' if dynamic else "None"
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
# exports never need gradients
torch.set_grad_enabled(False)
dummy_input = torch.randn({batch}, 3, imgsz, imgsz)
torch.onnx.export(
    model,
    dummy_input,
    output_path,
    opset_version=opset,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={dynamic_axes},
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
    return _run_export(
        directory,
        config,
        python_path,
        bin_path,
        model,
        script_content,
        "_export_torchvision_classifier.py",
        no_uv_cache=no_uv_cache,
        verbose=verbose,
    )


def export_torchvision_segmenter(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,
    model: str,
    opset: int,
    imgsz: int | None = None,
    *,
    batch: int = 1,
    dynamic: bool = False,
    no_cache: bool | None = None,  # noqa: ARG001
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,  # noqa: ARG001
    verbose: bool | None = None,
) -> Path:
    _520 = 520
    if imgsz is None:
        imgsz = _520
    dynamic_axes = '{"input": {0: "batch_size"}, "logits": {0: "batch_size"}}' if dynamic else "None"
    # the models return {"out", "aux"}; only "out" (already at input size) is exported
    script_content = f"""\
import torch
import torchvision.models.segmentation as models
import onnx
import onnxslim

model_name = "{config["name"]}"
opset = {opset}
imgsz = {imgsz}
output_path = model_name + ".onnx"


class Logits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]


model = Logits(getattr(models, model_name)(weights="DEFAULT"))
model.eval()
# exports never need gradients
torch.set_grad_enabled(False)
dummy_input = torch.randn({batch}, 3, imgsz, imgsz)
torch.onnx.export(
    model,
    dummy_input,
    output_path,
    opset_version=opset,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={dynamic_axes},
    dynamo=False,
)

slim_model = onnxslim.slim(output_path)
onnx.save(slim_model, output_path)
"""
    return _run_export(
        directory,
        config,
        python_path,
        bin_path,
        model,
        script_content,
        "_export_torchvision_segmenter.py",
        no_uv_cache=no_uv_cache,
        verbose=verbose,
    )


def _run_export(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,
    model: str,
    script_content: str,
    script_name: str,
    *,
    no_uv_cache: bool | None,
    verbose: bool | None,
) -> Path:
    # both tasks share the torchvision export requirements
    run_uv_pip_install(
        directory,
        bin_path.parent,
        "torchvision_classifier",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    script_path = directory / script_name
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
