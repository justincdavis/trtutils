# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np

with contextlib.suppress(ImportError):
    import tensorrt as trt

if TYPE_CHECKING:
    from typing_extensions import Self


def _make_plugin_field(name: str, value: int | float | list[int] | list[float]) -> "trt.PluginField":
    """Create a TensorRT PluginField with appropriate dtype inferred from value."""
    if isinstance(value, (list, tuple)):
        # Infer dtype from first element; default to float32 for floats, int32 for ints
        dtype = trt.PluginFieldType.FLOAT32 if isinstance(value[0], float) else trt.PluginFieldType.INT32
        arr = np.array(value, dtype=np.float32 if dtype == trt.PluginFieldType.FLOAT32 else np.int32)
    elif isinstance(value, float):
        dtype = trt.PluginFieldType.FLOAT32
        arr = np.array([value], dtype=np.float32)
    else:
        dtype = trt.PluginFieldType.INT32
        arr = np.array([value], dtype=np.int32)
    return trt.PluginField(name, arr, dtype)


def _get_output_like_yolo(network: "trt.INetworkDefinition") -> "trt.ITensor":
    """Find a YOLO-like output tensor shaped (N, 84, M) or (N, M, 84)."""
    for i in range(network.num_outputs):
        out = network.get_output(i)
        dims = out.shape
        if len(dims) != 3:
            continue
        if dims[1] == 84 or dims[2] == 84:
            return out
    raise RuntimeError("Could not find YOLO-like output with 3D shape containing channel size 84.")


def _transpose_nc_to_nlc(network: "trt.INetworkDefinition", tensor: "trt.ITensor") -> "trt.ITensor":
    """Ensure tensor is (N, num_boxes, C=84). If it's (N, 84, num_boxes), transpose to (0, 2, 1)."""
    dims = tensor.shape
    if dims[1] == 84 and dims[2] != 84:
        shuffle = network.add_shuffle(tensor)
        shuffle.first_transpose = (0, 2, 1)
        return shuffle.get_output(0)
    return tensor


def _slice_dynamic(
    network: "trt.INetworkDefinition",
    tensor: "trt.ITensor",
    start_vals: list[int],
    size_last_dim: int,
) -> "trt.ITensor":
    """Slice tensor dynamically on the last dimension using shape tensors.

    Keeps batch and num_boxes from input, and sets last dim to provided constant size.
    start_vals is a list of length 3 for (start_b, start_boxes, start_c).
    """
    assert len(start_vals) == 3

    # Build dynamic size [B, num_boxes, size_last_dim]
    shape_layer = network.add_shape(tensor)
    shape_tensor = shape_layer.get_output(0)  # int32 [3]

    # Gather batch dim (0) and num_boxes dim (1)
    def gather_dim(dim_index: int) -> "trt.ITensor":
        idx_const = network.add_constant((1,), np.array([dim_index], dtype=np.int32)).get_output(0)
        return network.add_gather(shape_tensor, idx_const, 0).get_output(0)

    b_dim = gather_dim(0)
    nb_dim = gather_dim(1)
    last_const = network.add_constant((1,), np.array([size_last_dim], dtype=np.int32)).get_output(0)

    size_concat = network.add_concatenation([b_dim, nb_dim, last_const])
    size_concat.axis = 0
    dyn_size = size_concat.get_output(0)

    # Start and stride tensors
    start_const = network.add_constant((3,), np.array(start_vals, dtype=np.int32)).get_output(0)
    stride_const = network.add_constant((3,), np.array([1, 1, 1], dtype=np.int32)).get_output(0)

    # Create slice with dummy dims then set dynamic inputs
    slice_layer = network.add_slice(tensor, (0, 0, 0), (1, 1, size_last_dim), (1, 1, 1))
    slice_layer.set_input(1, start_const)
    slice_layer.set_input(2, dyn_size)
    slice_layer.set_input(3, stride_const)
    return slice_layer.get_output(0)


def yolo_efficient_nms_hook(
    num_classes: int = 80,
    score_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_output_boxes: int = 100,
    class_agnostic: bool = False,
    box_coding: str | int = "CENTER_SIZE",
) -> "trt.INetworkDefinition":
    """
    Create a hook that injects EfficientNMS into a YOLO network with raw output shaped
    like (N, 84, num_boxes) or (N, num_boxes, 84).

    - Interprets first 4 channels as box coordinates and the remaining as class scores
    - Uses EfficientNMS plugin if available; falls back to BatchedNMSDynamic otherwise
    - Replaces raw network outputs with NMS outputs: num_dets, det_boxes, det_scores, det_classes

    Parameters can be customized to match dataset/model specifics.
    """

    def _hook(network: "trt.INetworkDefinition") -> "trt.INetworkDefinition":
        # Locate YOLO raw output and unmark it
        raw_out = _get_output_like_yolo(network)
        # Keep a copy of all outputs so we can unmark later
        old_outputs = [network.get_output(i) for i in range(network.num_outputs)]

        # Ensure layout is (N, num_boxes, 84)
        yolo_out = _transpose_nc_to_nlc(network, raw_out)

        # Boxes: first 4 channels; Scores: last num_classes channels
        boxes = _slice_dynamic(network, yolo_out, [0, 0, 0], 4)
        scores = _slice_dynamic(network, yolo_out, [0, 0, 4], num_classes)

        # Try EfficientNMS_TRT first
        registry = trt.get_plugin_registry()
        eff_creator = registry.get_plugin_creator("EfficientNMS_TRT", "1", "")

        if eff_creator is not None:
            # Map box_coding to int
            coding_map = {"CORNER": 0, "CENTER_SIZE": 1, 0: 0, 1: 1}
            box_coding_int = coding_map.get(box_coding, 1)

            fields = [
                _make_plugin_field("background_class", -1),
                _make_plugin_field("max_output_boxes", int(max_output_boxes)),
                _make_plugin_field("score_threshold", float(score_threshold)),
                _make_plugin_field("iou_threshold", float(iou_threshold)),
                _make_plugin_field("box_coding", int(box_coding_int)),
                _make_plugin_field("score_activation", 0),
                _make_plugin_field("class_agnostic", 1 if class_agnostic else 0),
            ]
            pfc = trt.PluginFieldCollection(fields)
            plugin = eff_creator.create_plugin("efficient_nms", pfc)
            plugin_layer = network.add_plugin_v2([boxes, scores], plugin)

            num_dets, det_boxes, det_scores, det_classes = (
                plugin_layer.get_output(0),
                plugin_layer.get_output(1),
                plugin_layer.get_output(2),
                plugin_layer.get_output(3),
            )
        else:
            # Fallback to BatchedNMSDynamic_TRT
            batched_creator = registry.get_plugin_creator("BatchedNMSDynamic_TRT", "1", "")
            if batched_creator is None:
                raise RuntimeError(
                    "Neither EfficientNMS_TRT nor BatchedNMSDynamic_TRT plugins are available."
                )

            # BatchedNMS expects boxes shaped [N, num_boxes, num_loc_classes, 4]
            # with shareLocation = 1 -> num_loc_classes = 1
            add_dim = network.add_shuffle(boxes)
            add_dim.reshape_dims = (0, 0, 1, 4)  # insert singleton dim at axis 2
            boxes_batched = add_dim.get_output(0)

            # Heuristic topK: use a large constant; keepTopK is max_output_boxes
            fields = [
                _make_plugin_field("shareLocation", 1),
                _make_plugin_field("backgroundLabelId", -1),
                _make_plugin_field("numClasses", int(num_classes)),
                _make_plugin_field("topK", 4096),
                _make_plugin_field("keepTopK", int(max_output_boxes)),
                _make_plugin_field("scoreThreshold", float(score_threshold)),
                _make_plugin_field("iouThreshold", float(iou_threshold)),
                _make_plugin_field("isNormalized", 1),
                _make_plugin_field("clipBoxes", 0),
            ]
            pfc = trt.PluginFieldCollection(fields)
            plugin = batched_creator.create_plugin("batched_nms", pfc)
            plugin_layer = network.add_plugin_v2([boxes_batched, scores], plugin)

            # BatchedNMS outputs: nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_indices
            det_boxes = plugin_layer.get_output(0)
            det_scores = plugin_layer.get_output(1)
            det_classes = plugin_layer.get_output(2)

            # Create num_dets by counting non-zero scores up to keepTopK
            # Approximation: use a shuffle to get shape [N,1] for num_dets is not directly supported.
            # Instead, expose indices tensor shape [N, keepTopK] as num_dets proxy is not ideal.
            # Many consumers rely on three outputs; we will add a constant -1 tensor for num_dets.
            num_dets = network.add_constant((1, 1), np.array([[0]], dtype=np.int32)).get_output(0)

        # Unmark old outputs
        for t in old_outputs:
            network.unmark_output(t)

        # Mark new outputs
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

