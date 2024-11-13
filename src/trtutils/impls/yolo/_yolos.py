# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

from ._yolo import YOLO

if TYPE_CHECKING:
    from pathlib import Path

    from typing_extensions import Self


class YOLOX(YOLO):
    """Implementation of YOLOX."""

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 255),
        preprocessor: str = "cuda",
        resize_method: str = "letterbox",
        conf_thres: float = 0.1,
        nms_iou_thres: float = 0.5,
        *,
        warmup: bool | None = None,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
    ) -> None:
        super().__init__(
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_range=input_range,
            preprocessor=preprocessor,
            resize_method=resize_method,
            conf_thres=conf_thres,
            nms_iou_thres=nms_iou_thres,
            warmup=warmup,
            extra_nms=extra_nms,
            agnostic_nms=agnostic_nms,
        )


class YOLO7(YOLO):
    """Implementation of YOLO7."""

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 1),
        preprocessor: str = "cuda",
        resize_method: str = "letterbox",
        conf_thres: float = 0.1,
        nms_iou_thres: float = 0.5,
        *,
        warmup: bool | None = None,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
    ) -> None:
        super().__init__(
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_range=input_range,
            preprocessor=preprocessor,
            resize_method=resize_method,
            conf_thres=conf_thres,
            nms_iou_thres=nms_iou_thres,
            warmup=warmup,
            extra_nms=extra_nms,
            agnostic_nms=agnostic_nms,
        )


class YOLO8(YOLO):
    """Implementation of YOLO8."""

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 1),
        preprocessor: str = "cuda",
        resize_method: str = "letterbox",
        conf_thres: float = 0.1,
        nms_iou_thres: float = 0.5,
        *,
        warmup: bool | None = None,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
    ) -> None:
        super().__init__(
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_range=input_range,
            preprocessor=preprocessor,
            resize_method=resize_method,
            conf_thres=conf_thres,
            nms_iou_thres=nms_iou_thres,
            warmup=warmup,
            extra_nms=extra_nms,
            agnostic_nms=agnostic_nms,
        )


class YOLO9(YOLO):
    """Implementation of YOLO9."""

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 1),
        preprocessor: str = "cuda",
        resize_method: str = "letterbox",
        conf_thres: float = 0.1,
        nms_iou_thres: float = 0.5,
        *,
        warmup: bool | None = None,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
    ) -> None:
        super().__init__(
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_range=input_range,
            preprocessor=preprocessor,
            resize_method=resize_method,
            conf_thres=conf_thres,
            nms_iou_thres=nms_iou_thres,
            warmup=warmup,
            extra_nms=extra_nms,
            agnostic_nms=agnostic_nms,
        )


class YOLO10(YOLO):
    """Implementation of YOLO10."""

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 1),
        preprocessor: str = "cuda",
        resize_method: str = "letterbox",
        conf_thres: float = 0.1,
        nms_iou_thres: float = 0.5,
        *,
        warmup: bool | None = None,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
    ) -> None:
        super().__init__(
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_range=input_range,
            preprocessor=preprocessor,
            resize_method=resize_method,
            conf_thres=conf_thres,
            nms_iou_thres=nms_iou_thres,
            warmup=warmup,
            extra_nms=extra_nms,
            agnostic_nms=agnostic_nms,
        )
