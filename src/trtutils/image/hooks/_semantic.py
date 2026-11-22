# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from trtutils.compat._libs import trt

if TYPE_CHECKING:
    from collections.abc import Callable

_LOGITS_RANK = 4


def semantic_argmax_hook(
    height: int,
    width: int,
) -> Callable[[trt.INetworkDefinition], trt.INetworkDefinition]:
    """
    Make a hook that replaces the logits output of a segmentation network with a class map.

    The ``(B, C, h, w)`` logits are resized bilinearly (half-pixel, as PyTorch's
    ``align_corners=False``) to the input size when they are smaller, then reduced over
    the class axis with a TopK(k=1). The int32 indices ``(B, 1, height, width)`` become the
    only output, named ``class_map``. Copying the class map instead of the logits cuts the
    device-to-host transfer by about the number of classes.

    Parameters
    ----------
    height : int
        The input image height, the class map's height.
    width : int
        The input image width, the class map's width.

    Returns
    -------
    Callable[[trt.INetworkDefinition], trt.INetworkDefinition]
        A hook that can be used to modify a network.

    Raises
    ------
    ValueError
        Raised by the hook when the network does not have a single 4D output.

    """

    def _hook(network: trt.INetworkDefinition) -> trt.INetworkDefinition:
        if network.num_outputs != 1 or len(network.get_output(0).shape) != _LOGITS_RANK:
            err_msg = "The argmax hook needs a network with a single (B, C, h, w) logits output"
            raise ValueError(err_msg)
        logits = network.get_output(0)
        scores = logits
        if tuple(logits.shape[2:]) != (height, width):
            resize = network.add_resize(logits)
            resize.set_input(1, _resized_shape(network, logits, height, width))
            resize.resize_mode = trt.InterpolationMode.LINEAR
            resize.coordinate_transformation = trt.ResizeCoordinateTransformation.HALF_PIXEL
            scores = resize.get_output(0)
        topk = network.add_topk(scores, trt.TopKOperation.MAX, 1, 1 << 1)
        class_map = topk.get_output(1)
        network.unmark_output(logits)
        class_map.name = "class_map"
        network.mark_output(class_map)
        return network

    return _hook


def _resized_shape(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    height: int,
    width: int,
) -> trt.ITensor:
    # (B, C) from the runtime shape so a dynamic batch works, then (height, width)
    shape = network.add_shape(tensor).get_output(0)
    batch_channels = network.add_slice(shape, (0,), (2,), (1,)).get_output(0)
    # trt 10 shape tensors are int64, trt 8.6 int32
    dtype = np.int32 if shape.dtype == trt.DataType.INT32 else np.int64
    size = network.add_constant((2,), np.array([height, width], dtype=dtype)).get_output(0)
    return network.add_concatenation([batch_channels, size]).get_output(0)
