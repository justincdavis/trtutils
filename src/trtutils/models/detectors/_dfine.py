# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from trtutils.image._detector import Detector
from trtutils.image._schema import InputSchema, OutputSchema
from trtutils.models._model import Model

from ._archs import DETR

if TYPE_CHECKING:
    from pathlib import Path

    from typing_extensions import Self


class DFINE(DETR, Model):
    """Alias of DETR with default args for D-FINE."""

    _model_type = "dfine"
    _friendly_name = "D-FINE"
    _default_imgsz = 640
    _valid_imgszs: ClassVar[list[int]] = [640]
    _input_tensors: ClassVar[list[tuple[str, str]]] = [
        ("images", "image"),
        ("orig_target_sizes", "size"),
    ]

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
            input_schema=InputSchema.RT_DETR,
            output_schema=OutputSchema.DETR_LBS,
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
