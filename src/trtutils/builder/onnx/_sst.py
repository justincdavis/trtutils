# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Move scale, channel swap, and HWC->CHW into an ONNX graph."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

if TYPE_CHECKING:
    from pathlib import Path

_IMAGE_RANK = 4
_RGB_CHANNELS = 3


def embed_sst(
    source: Path | str,
    dest: Path | str,
    image_name: str,
    *,
    scale: float,
    height: int,
    width: int,
) -> None:
    """
    Prepend scale, BGR-to-RGB, and HWC-to-CHW in front of an image network.

    The image input keeps its name and becomes uint8 ``(batch, height, width, 3)``,
    the layout a letterbox kernel already writes. Nodes that used to read that
    input now read the NCHW transpose. TensorRT folds the cast, scale, swap,
    and transpose into the first convolution.

    Parameters
    ----------
    source : Path | str
        ONNX model whose image input is float ``(batch, 3, height, width)``.
    dest : Path | str
        Where to write the rewritten model.
    image_name : str
        Name of that image input.
    scale : float
        Multiplier applied after the cast to float. ``1/255`` for the usual
        YOLO range ``(0, 1)``; ``1`` when the network wants ``(0, 255)``.
    height : int
        Network input height baked into the new uint8 input.
    width : int
        Network input width baked into the new uint8 input.

    Raises
    ------
    ValueError
        If ``image_name`` is missing or is not a 4D tensor with 3 channels
        on axis 1.

    """
    model = onnx.load(source)
    graph = model.graph
    image = next((item for item in graph.input if item.name == image_name), None)
    if image is None:
        err_msg = f"ONNX model has no input {image_name!r}"
        raise ValueError(err_msg)
    dims = image.type.tensor_type.shape.dim
    if (
        len(dims) != _IMAGE_RANK
        or not dims[1].HasField("dim_value")
        or dims[1].dim_value != _RGB_CHANNELS
    ):
        err_msg = f"{image_name!r} must be (batch, 3, height, width), got rank {len(dims)}"
        raise ValueError(err_msg)

    new_input = helper.make_tensor_value_info(image_name, TensorProto.UINT8, None)
    new_dims = new_input.type.tensor_type.shape.dim
    new_dims.add().CopyFrom(dims[0])
    for value in (height, width, _RGB_CHANNELS):
        new_dims.add().dim_value = value

    kept = [item for item in graph.input if item.name != image_name]
    del graph.input[:]
    graph.input.extend([new_input, *kept])

    nchw = f"{image_name}_nchw"
    scale_name = f"{image_name}_scale"
    order_name = f"{image_name}_bgr_to_rgb"
    graph.initializer.extend(
        [
            numpy_helper.from_array(np.array([scale], np.float32), name=scale_name),
            numpy_helper.from_array(np.array([2, 1, 0], np.int64), name=order_name),
        ]
    )
    front = [
        helper.make_node("Cast", [image_name], [f"{image_name}_f"], to=TensorProto.FLOAT),
        helper.make_node("Mul", [f"{image_name}_f", scale_name], [f"{image_name}_scaled"]),
        helper.make_node(
            "Gather",
            [f"{image_name}_scaled", order_name],
            [f"{image_name}_rgb"],
            axis=3,
        ),
        helper.make_node("Transpose", [f"{image_name}_rgb"], [nchw], perm=[0, 3, 1, 2]),
    ]
    existing = list(graph.node)
    for node in existing:
        node.input[:] = [nchw if name == image_name else name for name in node.input]
    del graph.node[:]
    graph.node.extend([*front, *existing])
    onnx.save(model, dest)
