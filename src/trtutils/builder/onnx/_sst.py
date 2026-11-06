# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Move scale, channel swap, mean/std, and HWC->CHW into an ONNX graph."""

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
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
) -> None:
    """
    Prepend normalization and HWC-to-CHW in front of an image network.

    The image input keeps its name and becomes uint8 ``(batch, height, width, 3)``,
    the layout a resize kernel already writes. Nodes that used to read that input
    now read the NCHW result. TensorRT folds these pointwise ops into the first
    convolution. Resize stays outside the graph.

    Without ``mean`` and ``std`` the nodes are cast, multiply by ``scale``, swap
    BGR to RGB, and transpose. That is YOLO (``scale`` is ``1/255``) and YOLOX
    (``scale`` is ``1``). With ``mean`` and ``std`` the multiply is ``1/255``,
    then ``(rgb - mean) / std``, which is the ImageNet path used by classifiers,
    depth models, and the detectors that share that recipe.

    Parameters
    ----------
    source : Path | str
        ONNX model whose image input is float ``(batch, 3, height, width)``.
    dest : Path | str
        Where to write the rewritten model.
    image_name : str
        Name of that image input.
    scale : float
        Multiplier applied after the cast to float when ``mean`` is omitted.
    height : int
        Network input height baked into the new uint8 input.
    width : int
        Network input width baked into the new uint8 input.
    mean : tuple[float, float, float] | None
        Per-channel mean in RGB order, applied after the channel swap.
    std : tuple[float, float, float] | None
        Per-channel standard deviation in RGB order.

    Raises
    ------
    ValueError
        If ``image_name`` is missing or is not a 4D tensor with 3 channels
        on axis 1, or only one of ``mean`` and ``std`` is set.

    """
    if (mean is None) != (std is None):
        err_msg = "mean and std must both be set or both be omitted"
        raise ValueError(err_msg)
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
    applied = (1.0 / 255.0) if mean is not None else scale
    graph.initializer.extend(
        [
            numpy_helper.from_array(np.array([applied], np.float32), name=scale_name),
            numpy_helper.from_array(np.array([2, 1, 0], np.int64), name=order_name),
        ]
    )
    scaled = f"{image_name}_scaled"
    rgb = f"{image_name}_rgb"
    front = [
        helper.make_node("Cast", [image_name], [f"{image_name}_f"], to=TensorProto.FLOAT),
        helper.make_node("Mul", [f"{image_name}_f", scale_name], [scaled]),
        helper.make_node("Gather", [scaled, order_name], [rgb], axis=3),
    ]
    transpose_in = rgb
    if mean is not None and std is not None:
        mean_name = f"{image_name}_mean"
        std_name = f"{image_name}_std"
        for name, values in ((mean_name, mean), (std_name, std)):
            channels = np.array(values, np.float32).reshape(1, 1, 1, _RGB_CHANNELS)
            graph.initializer.append(numpy_helper.from_array(channels, name=name))
        centered = f"{image_name}_centered"
        front.append(helper.make_node("Sub", [rgb, mean_name], [centered]))
        transpose_in = f"{image_name}_norm"
        front.append(helper.make_node("Div", [centered, std_name], [transpose_in]))
    front.append(helper.make_node("Transpose", [transpose_in], [nchw], perm=[0, 3, 1, 2]))
    existing = list(graph.node)
    for node in existing:
        node.input[:] = [nchw if name == image_name else name for name in node.input]
    del graph.node[:]
    graph.node.extend([*front, *existing])
    onnx.save(model, dest)
