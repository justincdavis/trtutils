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


def rtdetr_build_hook(
    *,
    onnx: Path | str,
    imgsz: int,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    top_k: int = 100,
    **_: Any,  # noqa: ANN401
) -> dict[str, Any]:
    """
    Build hook that adds EfficientNMS to an Ultralytics RT-DETR export.

    The Ultralytics exports (rtdetr-l, rtdetr-x) have a single ``output0`` head of normalized
    (cx, cy, w, h) boxes plus class scores. The boxes are scaled to pixels and passed through
    EfficientNMS. Exports from the original RT-DETR repository already decode their
    boxes, so they get no overrides.

    Parameters
    ----------
    onnx : Path | str
        Path to the ONNX model.
    imgsz : int
        Input image size, the factor that turns normalized boxes into pixels.
    conf_threshold : float
        Confidence threshold for EfficientNMS.
    iou_threshold : float
        IoU threshold for EfficientNMS.
    top_k : int
        Maximum number of detections to keep.

    Returns
    -------
    dict[str, Any]
        Build engine overrides containing hooks list, or an empty dict.

    """
    network, _, _, _ = read_onnx(onnx)
    if network.get_output(0).name != "output0":
        return {}
    return {
        "hooks": [
            yolo_efficient_nms_hook(
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                top_k=top_k,
                box_scale=imgsz,
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


def hands23_fp32_build_hook(*, onnx: Path | str, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
    """
    Build hook that forces FP32 for the proposal, box head and relation heads of Hands23.

    A full FP16 build flips the hand side (left/right) of many detections and adds spurious
    ones (seen on TensorRT 10.15, RTX 5080). Pinning every float layer from the proposal
    decoding (after the RPN head convolutions) through the z-head pair scores to FP32 fixes
    it. The backbone, FPN and RPN head convolutions, which hold nearly all the compute, stay
    FP16. A strongly typed build returns no overrides: those networks take precision from the
    ONNX graph, and ``build_engine`` rejects layer precision in that mode.

    Parameters
    ----------
    onnx : Path | str
        Path to the ONNX model.
    **kwargs : Any
        Extra arguments from ``Model.build``. ``strongly_typed`` is read here and still
        forwarded, because it is not a named parameter of this hook.

    Returns
    -------
    dict[str, Any]
        Build engine overrides containing the layer_precision list, or an empty dict.

    """
    # unnamed so Model.build does not consume strongly_typed before build_engine sees it
    if kwargs.get("strongly_typed"):
        return {}
    network, _, _, _ = read_onnx(onnx)
    names = [network.get_layer(idx).name for idx in range(network.num_layers)]
    start_name = "/proposal_generator/Transpose"
    if start_name not in names:
        LOG.warning("Hands23 FP32 hook: layer names not found, building the whole model in FP16")
        return {}
    start = names.index(start_name)
    # the first /Greater is the v_norm check mid-geometry; pair_probs is the z-head output
    stop = None
    for idx in range(start, network.num_layers):
        layer = network.get_layer(idx)
        for out_idx in range(layer.num_outputs):
            if layer.get_output(out_idx).name == "pair_probs":
                stop = idx
    if stop is None:
        LOG.warning("Hands23 FP32 hook: layer names not found, building the whole model in FP16")
        return {}
    layer_precision = []
    for idx in range(start, stop + 1):
        layer = network.get_layer(idx)
        if layer.num_outputs and layer.get_output(0).dtype == trt.DataType.FLOAT:
            layer_precision.append((idx, trt.DataType.FLOAT))
    return {"layer_precision": layer_precision, "prefer_precision_constraints": True}
