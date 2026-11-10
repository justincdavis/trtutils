# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Move channel swap, normalization, and HWC->CHW into an ONNX graph."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

_IMAGE_RANK = 4
_RGB_CHANNELS = 3


def embed_sst(
    source: Path | str,
    dest: Path | str,
    image_name: str,
    *,
    height: int,
    width: int,
    input_range: tuple[float, float] = (0.0, 1.0),
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
) -> None:
    """
    Prepend channel swap, normalization, and HWC-to-CHW in front of an image network.

    The image input keeps its name and position and becomes uint8
    ``(batch, height, width, 3)``, the layout a resize kernel already writes. Nodes that
    used to read that input now read the NCHW result. The new nodes cast, swap BGR to RGB,
    and apply the same per-channel affine as the image preprocessors: ``input_range``
    alone maps ``[0, 255]`` onto it, ``mean`` and ``std`` give ``(x / 255 - mean) / std``.
    TensorRT folds these pointwise ops into the first layer. Resize stays outside.

    Parameters
    ----------
    source : Path | str
        ONNX model whose image input is ``(batch, 3, height, width)``.
    dest : Path | str
        Where to write the rewritten model.
    image_name : str
        Name of that image input.
    height : int
        Network input height baked into the new uint8 input.
    width : int
        Network input width baked into the new uint8 input.
    input_range : tuple[float, float]
        Output range when ``mean`` and ``std`` are omitted. Default is ``(0.0, 1.0)``.
    mean : tuple[float, float, float] | None
        Per-channel mean in RGB order.
    std : tuple[float, float, float] | None
        Per-channel standard deviation in RGB order.

    Raises
    ------
    ValueError
        If ``image_name`` is missing or is not ``(batch, 3, height, width)``,
        or only one of ``mean`` and ``std`` is set.

    """
    if (mean is None) != (std is None):
        err_msg = "mean and std must both be set or both be omitted"
        raise ValueError(err_msg)
    model = onnx.load(source)
    graph = model.graph
    index = next((i for i, item in enumerate(graph.input) if item.name == image_name), None)
    if index is None:
        err_msg = f"ONNX model has no input {image_name!r}"
        raise ValueError(err_msg)
    dims = graph.input[index].type.tensor_type.shape.dim
    found = [dim.dim_value if dim.HasField("dim_value") else None for dim in dims]
    # dynamic height/width (None) take any size; static ones must match
    if (
        len(found) != _IMAGE_RANK
        or found[1] != _RGB_CHANNELS
        or found[2] not in (None, height)
        or found[3] not in (None, width)
    ):
        err_msg = f"{image_name!r} must be (batch, 3, {height}, {width}), got {found}"
        raise ValueError(err_msg)

    # same affine as the CPU and CUDA preprocessors
    if mean is not None and std is not None:
        std_arr = np.array(std, np.float32)
        scale = 1.0 / (255.0 * std_arr)
        offset = -np.array(mean, np.float32) / std_arr
    else:
        low, high = input_range
        scale = np.full(_RGB_CHANNELS, (high - low) / 255.0, np.float32)
        offset = np.full(_RGB_CHANNELS, low, np.float32)

    new_input = helper.make_tensor_value_info(image_name, TensorProto.UINT8, None)
    new_dims = new_input.type.tensor_type.shape.dim
    new_dims.add().CopyFrom(dims[0])
    for value in (height, width, _RGB_CHANNELS):
        new_dims.add().dim_value = value
    graph.input[index].CopyFrom(new_input)

    names = {
        step: f"{image_name}_{step}"
        for step in ("order", "scale", "offset", "rgb", "float", "scaled", "norm", "nchw")
    }
    graph.initializer.extend(
        [
            numpy_helper.from_array(np.array([2, 1, 0], np.int64), name=names["order"]),
            numpy_helper.from_array(scale, name=names["scale"]),
            numpy_helper.from_array(offset, name=names["offset"]),
        ]
    )
    front = [
        # TensorRT has no uint8 Gather, so cast first
        helper.make_node("Cast", [image_name], [names["float"]], to=TensorProto.FLOAT),
        helper.make_node("Gather", [names["float"], names["order"]], [names["rgb"]], axis=3),
        helper.make_node("Mul", [names["rgb"], names["scale"]], [names["scaled"]]),
        helper.make_node("Add", [names["scaled"], names["offset"]], [names["norm"]]),
        helper.make_node("Transpose", [names["norm"]], [names["nchw"]], perm=[0, 3, 1, 2]),
    ]
    existing = list(graph.node)
    for node in existing:
        node.input[:] = [names["nchw"] if name == image_name else name for name in node.input]
    del graph.node[:]
    graph.node.extend([*front, *existing])
    # models over the 2GB protobuf limit (ViT-H, HOI-DETR ViT-L) keep their weights outside
    large = model.ByteSize() >= onnx.checker.MAXIMUM_PROTOBUF
    location = f"{Path(dest).name}.data" if large else None
    onnx.save(model, dest, save_as_external_data=large, location=location)
