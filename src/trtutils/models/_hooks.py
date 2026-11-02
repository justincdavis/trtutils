# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Build hooks shared by the model wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from trtutils._log import LOG
from trtutils.builder._onnx import read_onnx
from trtutils.builder.hooks import yolo_efficient_nms_hook
from trtutils.compat._libs import trt

if TYPE_CHECKING:
    from pathlib import Path


def nms_build_hook(
    *,
    num_classes: int = 80,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    top_k: int = 100,
    **_: Any,  # noqa: ANN401
) -> dict[str, Any]:
    """
    Build hook that adds EfficientNMS to a YOLO network.

    Parameters
    ----------
    num_classes : int
        Number of classes for EfficientNMS.
    conf_threshold : float
        Confidence threshold for EfficientNMS.
    iou_threshold : float
        IoU threshold for EfficientNMS.
    top_k : int
        Maximum number of detections to keep.

    Returns
    -------
    dict[str, Any]
        Build engine overrides containing hooks list.

    """
    return {
        "hooks": [
            yolo_efficient_nms_hook(
                num_classes=num_classes,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                top_k=top_k,
            ),
        ],
    }


def depthwise_fp32_build_hook(*, onnx: Path | str, **_: Any) -> dict[str, Any]:  # noqa: ANN401
    """
    Build hook that forces FP32 for the depthwise convolutions.

    In FP16 the builder sometimes picks depthwise kernels that break the output (seen on
    TensorRT 10.15, RTX 5080): YOLOv10 loses large objects in about one build in seven,
    YOLOv13 detects nothing in about one in fifty. Depthwise convolutions are cheap, so
    FP32 costs little.

    Parameters
    ----------
    onnx : Path | str
        Path to the ONNX model.

    Returns
    -------
    dict[str, Any]
        Build engine overrides containing the layer_precision list.

    """
    network, _, _, _ = read_onnx(onnx)
    layer_precision = []
    for idx in range(network.num_layers):
        layer = network.get_layer(idx)
        if layer.type != trt.LayerType.CONVOLUTION:
            continue
        layer.__class__ = trt.IConvolutionLayer
        if layer.num_groups > 1:
            layer_precision.append((idx, trt.DataType.FLOAT))
    return {"layer_precision": layer_precision or None, "prefer_precision_constraints": True}


def hands23_fp32_build_hook(*, onnx: Path | str, **_: Any) -> dict[str, Any]:  # noqa: ANN401
    """
    Build hook that forces FP32 for the proposal, box head and relation heads of Hands23.

    A full FP16 build flips the hand side (left/right) of many detections and adds spurious
    ones (seen on TensorRT 10.15, RTX 5080). Pinning every float layer from the proposal
    decoding (after the RPN head convolutions) through the second-stage relation heads to FP32
    fixes it. The backbone, FPN and RPN head convolutions, which hold nearly all the compute,
    stay FP16.

    Parameters
    ----------
    onnx : Path | str
        Path to the ONNX model.

    Returns
    -------
    dict[str, Any]
        Build engine overrides containing the layer_precision list.

    """
    network, _, _, _ = read_onnx(onnx)
    names = [network.get_layer(idx).name for idx in range(network.num_layers)]
    if "/proposal_generator/Transpose" not in names or "/Greater" not in names:
        LOG.warning("Hands23 FP32 hook: layer names not found, building the whole model in FP16")
        return {}
    start = names.index("/proposal_generator/Transpose")
    stop = names.index("/Greater")
    layer_precision = []
    for idx in range(start, stop + 1):
        layer = network.get_layer(idx)
        if layer.num_outputs and layer.get_output(0).dtype == trt.DataType.FLOAT:
            layer_precision.append((idx, trt.DataType.FLOAT))
    return {"layer_precision": layer_precision, "prefer_precision_constraints": True}
