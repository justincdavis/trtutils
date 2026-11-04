# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from trtutils.image._detector import Detector
from trtutils.image._schema import InputSchema, OutputSchema
from trtutils.models._model import Model

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from typing_extensions import Self


class RFDETR(Detector, Model):
    """
    Alias of DETR with default args for RF-DETR.

    Resizes with a plain square resize (``resize_method="linear"``) like upstream RF-DETR, and
    returns the model's own class index (stock COCO weights use COCO category ids, horse is 19).
    """

    _model_type = "rfdetr"
    _friendly_name = "RF-DETR"
    _default_imgsz = 576
    _imgsz_divisor = 32
    _input_tensors: ClassVar[list[tuple[str, str]]] = [("input", "image")]

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 1),
        preprocessor: str = "trt",
        resize_method: str = "linear",
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
        postprocessor: str | None = None,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
        class_map: Mapping[int, int] | None = None,
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
            postprocessor=postprocessor,
            extra_nms=extra_nms,
            agnostic_nms=agnostic_nms,
            class_map=class_map,
            no_warn=no_warn,
            verbose=verbose,
        )
