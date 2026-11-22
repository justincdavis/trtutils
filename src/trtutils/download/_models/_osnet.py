# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

from trtutils.download._tools import (
    git_clone,
    normalize_imgsz,
    run_cmd,
    run_download,
    run_uv_pip_install,
)

if TYPE_CHECKING:
    from pathlib import Path


def export_osnet(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,
    model: str,
    opset: int,
    imgsz: int | tuple[int, int] | None = None,
    *,
    batch: int = 1,
    dynamic: bool = False,
    no_cache: bool | None = None,
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,  # noqa: ARG001
    verbose: bool | None = None,
) -> Path:
    # osnet is fully convolutional (global average pool), any (height, width) works
    size = normalize_imgsz(config["imgsz"] if imgsz is None else imgsz)
    height, width = (size, size) if isinstance(size, int) else size
    dynamic_axes = (
        '{"input": {0: "batch_size"}, "embedding": {0: "batch_size"}}' if dynamic else "None"
    )
    git_clone(
        "https://github.com/KaiyangZhou/deep-person-reid",
        directory,
        "f8cd150fdf77e8d9e1ed143b7f308c2c609ded50",
        no_cache=no_cache,
        verbose=verbose,
    )
    run_uv_pip_install(
        directory,
        bin_path.parent,
        "osnet",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    run_download(directory, config, python_path, no_cache=no_cache, verbose=verbose)
    output_name = f"{model}.onnx"
    script_content = f"""\
import importlib.util

import onnx
import onnxslim
import torch

output_path = "{output_name}"

# load only osnet.py: the torchreid package init pulls in its data / engine stack
spec = importlib.util.spec_from_file_location(
    "osnet", "deep-person-reid/torchreid/models/osnet.py"
)
osnet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(osnet)

state = torch.load("{config["name"]}.{config["extension"]}", map_location="cpu", weights_only=True)
# the classifier head is sized for the training identities; eval mode returns the
# 512-d feature before it, so the exported graph does not contain it
num_classes = state["classifier.weight"].shape[0]
model = getattr(osnet, "{config["arch"]}")(num_classes=num_classes, pretrained=False)
model.load_state_dict(state)
model.eval()
# exports never need gradients
torch.set_grad_enabled(False)
dummy_input = torch.randn({batch}, 3, {height}, {width})
torch.onnx.export(
    model,
    dummy_input,
    output_path,
    dynamo=False,
    opset_version={opset},
    input_names=["input"],
    output_names=["embedding"],
    dynamic_axes={dynamic_axes},
)

slim_model = onnxslim.slim(output_path)
onnx.save(slim_model, output_path)
"""
    script_path = directory / "_export_osnet.py"
    script_path.write_text(script_content)
    run_cmd(
        [python_path, str(script_path)],
        cwd=directory,
        verbose=verbose,
    )
    return directory / output_name
