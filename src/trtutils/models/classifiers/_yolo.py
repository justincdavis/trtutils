# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from trtutils.image._classifier import Classifier
from trtutils.models._model import Model

if TYPE_CHECKING:
    from pathlib import Path

    from typing_extensions import Self


class YOLOv8Cls(Classifier, Model):
    """
    Alias of Classifier with default args for YOLOv8-Cls.

    softmax defaults to False since the exported ONNX head already
    applies softmax internally; re-applying it would flatten the confidences.
    """

    _model_type = "yolov8_cls"
    _friendly_name = "YOLOv8-Cls"
    _default_imgsz = 224
    _input_tensors: ClassVar[list[tuple[str, str]]] = [("images", "image")]

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 1),
        preprocessor: str = "trt",
        resize_method: str = "linear",
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
        dla_core: int | None = None,
        device: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = None,
        softmax: bool = False,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        Classifier.__init__(
            self,
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_range=input_range,
            preprocessor=preprocessor,
            resize_method=resize_method,
            mean=mean,
            std=std,
            dla_core=dla_core,
            device=device,
            backend=backend,
            warmup=warmup,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
            cuda_graph=cuda_graph,
            softmax=softmax,
            no_warn=no_warn,
            verbose=verbose,
        )


class YOLOv11Cls(Classifier, Model):
    """
    Alias of Classifier with default args for YOLOv11-Cls.

    softmax defaults to False since the exported ONNX head already
    applies softmax internally; re-applying it would flatten the confidences.
    """

    _model_type = "yolov11_cls"
    _friendly_name = "YOLOv11-Cls"
    _default_imgsz = 224
    _input_tensors: ClassVar[list[tuple[str, str]]] = [("images", "image")]

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 1),
        preprocessor: str = "trt",
        resize_method: str = "linear",
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
        dla_core: int | None = None,
        device: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = None,
        softmax: bool = False,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        Classifier.__init__(
            self,
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_range=input_range,
            preprocessor=preprocessor,
            resize_method=resize_method,
            mean=mean,
            std=std,
            dla_core=dla_core,
            device=device,
            backend=backend,
            warmup=warmup,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
            cuda_graph=cuda_graph,
            softmax=softmax,
            no_warn=no_warn,
            verbose=verbose,
        )


class YOLOv26Cls(Classifier, Model):
    """
    Alias of Classifier with default args for YOLOv26-Cls.

    softmax defaults to False since the exported ONNX head already
    applies softmax internally; re-applying it would flatten the confidences.
    """

    _model_type = "yolov26_cls"
    _friendly_name = "YOLOv26-Cls"
    _default_imgsz = 224
    _input_tensors: ClassVar[list[tuple[str, str]]] = [("images", "image")]

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 1),
        preprocessor: str = "trt",
        resize_method: str = "linear",
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
        dla_core: int | None = None,
        device: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = None,
        softmax: bool = False,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        Classifier.__init__(
            self,
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_range=input_range,
            preprocessor=preprocessor,
            resize_method=resize_method,
            mean=mean,
            std=std,
            dla_core=dla_core,
            device=device,
            backend=backend,
            warmup=warmup,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
            cuda_graph=cuda_graph,
            softmax=softmax,
            no_warn=no_warn,
            verbose=verbose,
        )
