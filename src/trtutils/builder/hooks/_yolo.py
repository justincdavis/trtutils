# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np

with contextlib.suppress(ImportError):
    import tensorrt as trt

from trtutils._flags import FLAGS

if TYPE_CHECKING:
    from collections.abc import Callable

from ._common import make_plugin_field

_YOLO_LIKE_DIMS = 3


def _get_yolo_output_tensor(network: trt.INetworkDefinition, num_classes: int) -> trt.ITensor:
    for i in range(network.num_outputs):
        out = network.get_output(i)
        dims = out.shape
        if len(dims) != _YOLO_LIKE_DIMS:
            continue
        if dims[1] == num_classes or dims[2] == num_classes:
            return out
    err_msg = f"Could not find YOLO-like output with 3D shape containing channel size {num_classes}."
    raise RuntimeError(err_msg)


def _transpose_nc_to_nlc(
    network: trt.INetworkDefinition, tensor: trt.ITensor, num_classes: int
) -> trt.ITensor:
    dims = tensor.shape
    if dims[1] == num_classes and dims[2] != num_classes:
        shuffle = network.add_shuffle(tensor)
        shuffle.first_transpose = (0, 2, 1)
        return shuffle.get_output(0)
    return tensor


def _slice_dynamic(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    start_vals: list[int],
    size_last_dim: int,
) -> trt.ITensor:
    if len(start_vals) != _YOLO_LIKE_DIMS:
        err_msg = f"Internal error: expected start_vals to be a list of length {_YOLO_LIKE_DIMS}, got {len(start_vals)}."
        raise ValueError(err_msg)

    shape_layer = network.add_shape(tensor)
    shape_tensor = shape_layer.get_output(0)
    st_dtype = shape_tensor.dtype
    np_int_dtype = np.int64 if FLAGS.TRT_HAS_INT64 and st_dtype == trt.DataType.INT64 else np.int32

    def gather_dim(dim_index: int) -> trt.ITensor:
        idx_const = network.add_constant((1,), np.array([dim_index], dtype=np_int_dtype)).get_output(
            0
        )
        return network.add_gather(shape_tensor, idx_const, 0).get_output(0)

    b_dim = gather_dim(0)
    nb_dim = gather_dim(1)
    last_const = network.add_constant(
        (1,), np.array([size_last_dim], dtype=np_int_dtype)
    ).get_output(0)
    size_concat = network.add_concatenation([b_dim, nb_dim, last_const])
    size_concat.axis = 0
    dyn_size = size_concat.get_output(0)

    start_const = network.add_constant((3,), np.array(start_vals, dtype=np_int_dtype)).get_output(0)
    stride_const = network.add_constant((3,), np.array([1, 1, 1], dtype=np_int_dtype)).get_output(0)
    slice_layer = network.add_slice(tensor, (0, 0, 0), (1, 1, size_last_dim), (1, 1, 1))
    slice_layer.set_input(1, start_const)
    slice_layer.set_input(2, dyn_size)
    slice_layer.set_input(3, stride_const)

    return slice_layer.get_output(0)


def yolo_efficient_nms_hook(
    num_classes: int = 80,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    top_k: int = 100,
    box_coding: str = "center_size",
    *,
    class_agnostic: bool | None = None,
) -> Callable[[trt.INetworkDefinition], trt.INetworkDefinition]:
    """
    Create a hook to add EfficientNMS_TRT plugin to YOLO-like output network.

    Expects a network with output shaped (N, num_classes, num_boxes) or (N, num_boxes, num_classes).

    - Interprets first 4 channels as box coordinates and the remaining as class scores
    - Supports outputs with or without an explicit objectness channel (4 + num_classes) or (4 + 1 + num_classes)
    - Replaces raw network outputs with NMS outputs: num_dets, det_boxes, det_scores, det_classes

    Parameters
    ----------
    num_classes : int, optional
        Number of classes in the dataset.
        Default is 80.
    conf_threshold : float, optional
        Confidence threshold for filtering boxes.
        Default is 0.25.
    iou_threshold : float, optional
        IoU threshold for NMS.
        Default is 0.5.
    top_k : int, optional
        Number of top detections to keep.
        Default is 100.
    class_agnostic : bool, optional
        Whether to use class-agnostic NMS.
        Default is False.
    box_coding : str, optional
        Coding of the bounding boxes.
        Default is "center_size".

    Returns
    -------
    Callable[[trt.INetworkDefinition], trt.INetworkDefinition]
        A hook that can be used to modify a network.

    """

    def _hook(network: trt.INetworkDefinition) -> trt.INetworkDefinition:
        # assess if the network is YOLOv10, in which case no processing needed
        if network.get_output(0).shape == (1, 300, 6):
            return network

        # get the number of channels for the output
        # two cases:
        # 1. YOLOv8 style output: (N, num_classes + bbox_dim, num_boxes)
        # 2. YOLOX style output: (N, num_boxes, num_classes + bbox_dim + objectness)
        channels_without_obj = num_classes + 4
        channels_with_obj = num_classes + 5

        # resolve which case we are using
        try:
            raw_out = _get_yolo_output_tensor(network, channels_without_obj)
            channel_count = channels_without_obj
        except RuntimeError:
            raw_out = _get_yolo_output_tensor(network, channels_with_obj)
            channel_count = channels_with_obj

        # get the outputs, transpose if needed so we have the following format:
        # (N, num_boxes, num_classes + bbox_dim + objectness (if used))
        old_outputs = [network.get_output(i) for i in range(network.num_outputs)]
        yolo_out = _transpose_nc_to_nlc(network, raw_out, channel_count)
        boxes = _slice_dynamic(network, yolo_out, [0, 0, 0], 4)

        # assess if objectness is present
        # objectness has dimension 4 for bboxes, 1 for objectness, and num_classes for classes
        # if objectness is present, multiply it with class scores
        # if objectness is not present, use class scores directly
        if channel_count == channels_with_obj:
            obj_score = _slice_dynamic(network, yolo_out, [0, 0, 4], 1)
            class_scores = _slice_dynamic(network, yolo_out, [0, 0, 5], num_classes)
            scores = network.add_elementwise(
                obj_score, class_scores, trt.ElementWiseOperation.PROD
            ).get_output(0)
        else:
            scores = _slice_dynamic(network, yolo_out, [0, 0, 4], num_classes)

        # create the efficient nms plugin instance
        registry = trt.get_plugin_registry()
        eff_creator = registry.get_plugin_creator("EfficientNMS_TRT", "1", "")
        if eff_creator is None:
            err_msg = "EfficientNMS_TRT plugin not found."
            raise RuntimeError(err_msg)

        # setup the plugin fields of the efficient nms plugin
        box_coding_int = 0 if box_coding == "corner" else 1
        fields = [
            make_plugin_field("background_class", -1),
            make_plugin_field("max_output_boxes", int(top_k)),
            make_plugin_field("score_threshold", float(conf_threshold)),
            make_plugin_field("iou_threshold", float(iou_threshold)),
            make_plugin_field("box_coding", int(box_coding_int)),
            make_plugin_field("score_activation", 0),
            make_plugin_field("class_agnostic", 1 if class_agnostic else 0),
        ]
        pfc = trt.PluginFieldCollection(fields)
        plugin = eff_creator.create_plugin("efficient_nms", pfc)
        plugin_layer = network.add_plugin_v2([boxes, scores], plugin)

        # get outputs of the plugin
        num_dets, det_boxes, det_scores, det_classes = (
            plugin_layer.get_output(0),
            plugin_layer.get_output(1),
            plugin_layer.get_output(2),
            plugin_layer.get_output(3),
        )

        # unmark outputs of the old network
        for t in old_outputs:
            network.unmark_output(t)

        # mark outputs of efficient nms as the new outputs
        num_dets.name = "num_dets"
        det_boxes.name = "det_boxes"
        det_scores.name = "det_scores"
        det_classes.name = "det_classes"
        network.mark_output(num_dets)
        network.mark_output(det_boxes)
        network.mark_output(det_scores)
        network.mark_output(det_classes)

        return network

    return _hook
