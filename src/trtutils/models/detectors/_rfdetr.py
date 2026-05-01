# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from trtutils.compat._libs import trt
from trtutils.image._detector import Detector
from trtutils.image._schema import InputSchema, OutputSchema
from trtutils.inspect._onnx import inspect_onnx_layers
from trtutils.models._model import Model

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from typing_extensions import Self


def rfdetr_precision_build_hook(*, onnx: Path | str, **_: Any) -> dict[str, Any]:  # noqa: ANN401
    """
    Build hook that forces FP32 for ReduceMean and Downsample layers in RF-DETR.

    Parameters
    ----------
    onnx : Path | str
        Path to the ONNX model.

    Returns
    -------
    dict[str, Any]
        Build engine overrides containing layer_precision list.

    """
    layer_info = inspect_onnx_layers(onnx, verbose=False)
    layer_precision = []
    for info in layer_info:
        lower_name = info.name.lower()
        if "reducemean" in lower_name or "downsample" in lower_name:
            layer_precision.append((info.index, trt.DataType.FLOAT))
    return {"layer_precision": layer_precision or None}


class RFDETR(Detector, Model):
    """Alias of DETR with default args for RF-DETR."""

    _model_type = "rfdetr"
    _friendly_name = "RF-DETR"
    _default_imgsz = 576
    _imgsz_divisor = 32
    _input_tensors: ClassVar[list[tuple[str, str]]] = [("input", "image")]
    _build_hooks: ClassVar[list[Callable[..., dict[str, Any]]]] = [
        rfdetr_precision_build_hook,
    ]

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 1),
        preprocessor: str = "trt",
        resize_method: str = "letterbox",
        conf_thres: float = 0.1,
        nms_iou_thres: float = 0.5,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        dla_core: int | None = None,
        device: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = None,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        Detector.__init__(
            self,
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_range=input_range,
            preprocessor=preprocessor,
            resize_method=resize_method,
            conf_thres=conf_thres,
            nms_iou_thres=nms_iou_thres,
            mean=mean,
            std=std,
            input_schema=InputSchema.RF_DETR,
            output_schema=OutputSchema.RF_DETR,
            dla_core=dla_core,
            device=device,
            backend=backend,
            warmup=warmup,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
            cuda_graph=cuda_graph,
            extra_nms=extra_nms,
            agnostic_nms=agnostic_nms,
            no_warn=no_warn,
            verbose=verbose,
        )
