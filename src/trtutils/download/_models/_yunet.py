# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import urlsplit

from trtutils.download._tools import _640, handle_imgsz, run_cmd, run_download, run_uv_pip_install

# the coarsest stride of the heads, the input height and width must divide by it
_YUNET_DIVISOR = 32


def export_yunet(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,
    model: str,
    opset: int,  # noqa: ARG001
    imgsz: int | None = None,
    *,
    no_cache: bool | None = None,
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,  # noqa: ARG001
    verbose: bool | None = None,
) -> Path:
    # OpenCV Zoo ships YuNet as ready ONNX (opset 11, batch 1, symbolic height and width),
    # so there is nothing to export: download, check and patch the graph
    imgsz = handle_imgsz(imgsz, _640, "YuNet", adjust_div=_YUNET_DIVISOR)
    run_download(directory, config, python_path, no_cache=no_cache, verbose=verbose)
    source = directory / Path(urlsplit(config["url"]).path).name
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != config["sha256"]:
        err_msg = f"{source.name} has sha256 {digest}, expected {config['sha256']}"
        raise RuntimeError(err_msg)
    run_uv_pip_install(directory, bin_path.parent, "yunet", no_cache=no_uv_cache, verbose=verbose)

    onnx_name = f"{model}.onnx"
    # 1. yunet was trained on bgr images, a Gather flips the channels so the model takes
    #    rgb like the other models
    # 2. fix height and width to imgsz, and give every head its static number of cells
    #    (the upstream file names all of them by one symbolic dim)
    program = f"""
import numpy as np
import onnx
from onnx import helper, numpy_helper

S = {imgsz}
m = onnx.load("{source.name}")
g = m.graph
for node in g.node:
    for i, name in enumerate(node.input):
        if name == "input":
            node.input[i] = "input_bgr"
g.initializer.append(numpy_helper.from_array(np.array([2, 1, 0], dtype=np.int64), "rgb_to_bgr"))
g.node.insert(0, helper.make_node("Gather", ["input", "rgb_to_bgr"], ["input_bgr"], axis=1))
g.input[0].CopyFrom(helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3, S, S]))
for out in g.output:
    stride = int(out.name.split("_")[1])
    channels = out.type.tensor_type.shape.dim[2].dim_value
    shape = [1, (S // stride) ** 2, channels]
    out.CopyFrom(helper.make_tensor_value_info(out.name, onnx.TensorProto.FLOAT, shape))
onnx.checker.check_model(m, full_check=True)
onnx.save(m, "{onnx_name}")
"""
    run_cmd([python_path, "-c", program], cwd=directory, verbose=verbose)
    return directory / onnx_name
