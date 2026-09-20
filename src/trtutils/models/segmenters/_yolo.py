# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from trtutils.image._schema import InputSchema
from trtutils.image._segmenter import Segmenter
from trtutils.models._model import Model

if TYPE_CHECKING:
    from pathlib import Path

    from typing_extensions import Self


class YOLOv8Seg(Segmenter, Model):
    """Alias of Segmenter with default args for YOLOv8-Seg."""

    _model_type = "yolov8_seg"
    _friendly_name = "YOLOv8-Seg"
    _default_imgsz = 640
    _input_tensors: ClassVar[list[tuple[str, str]]] = [("images", "image")]

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 1),
        preprocessor: str = "trt",
        resize_method: str = "letterbox",
        conf_thres: float = 0.1,
        nms_iou_thres: float = 0.5,
        mask_thres: float = 0.5,
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
        Segmenter.__init__(
            self,
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_range=input_range,
            preprocessor=preprocessor,
            resize_method=resize_method,
            conf_thres=conf_thres,
            nms_iou_thres=nms_iou_thres,
            mask_thres=mask_thres,
            input_schema=InputSchema.YOLO,
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


class YOLOv11Seg(Segmenter, Model):
    """Alias of Segmenter with default args for YOLOv11-Seg."""

    _model_type = "yolov11_seg"
    _friendly_name = "YOLOv11-Seg"
    _default_imgsz = 640
    _input_tensors: ClassVar[list[tuple[str, str]]] = [("images", "image")]

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 1),
        preprocessor: str = "trt",
        resize_method: str = "letterbox",
        conf_thres: float = 0.1,
        nms_iou_thres: float = 0.5,
        mask_thres: float = 0.5,
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
        Segmenter.__init__(
            self,
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_range=input_range,
            preprocessor=preprocessor,
            resize_method=resize_method,
            conf_thres=conf_thres,
            nms_iou_thres=nms_iou_thres,
            mask_thres=mask_thres,
            input_schema=InputSchema.YOLO,
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


class YOLOv26Seg(Segmenter, Model):
    """Alias of Segmenter with default args for YOLOv26-Seg."""

    _model_type = "yolov26_seg"
    _friendly_name = "YOLOv26-Seg"
    _default_imgsz = 640
    _input_tensors: ClassVar[list[tuple[str, str]]] = [("images", "image")]

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 1),
        preprocessor: str = "trt",
        resize_method: str = "letterbox",
        conf_thres: float = 0.1,
        nms_iou_thres: float = 0.5,
        mask_thres: float = 0.5,
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
        Segmenter.__init__(
            self,
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_range=input_range,
            preprocessor=preprocessor,
            resize_method=resize_method,
            conf_thres=conf_thres,
            nms_iou_thres=nms_iou_thres,
            mask_thres=mask_thres,
            input_schema=InputSchema.YOLO,
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
