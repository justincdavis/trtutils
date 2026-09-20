# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from trtutils.image._hand_interaction import HandInteractionDetector
from trtutils.models._model import Model

if TYPE_CHECKING:
    from pathlib import Path

    from typing_extensions import Self


class Hands23(HandInteractionDetector, Model):
    """
    Alias of HandInteractionDetector with default args for Hands23.

    Normalization (BGR flip, mean/std) is baked into the exported ONNX, so
    the wrapper passes raw 0-255 RGB input straight through. A single
    conf_thres is used for all three classes, unlike the reference demo
    which uses per-class thresholds of 0.7/0.5/0.3. CUDA graphs default to
    off: the in-graph NMS yields data-dependent shapes, which TensorRT
    cannot capture.
    """

    _model_type = "hands23"
    _friendly_name = "Hands23"
    _default_imgsz = 800
    _imgsz_divisor = 32
    _input_tensors: ClassVar[list[tuple[str, str]]] = [("input", "image")]

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0.0, 255.0),
        preprocessor: str = "trt",
        resize_method: str = "letterbox",
        conf_thres: float = 0.3,
        pair_thres: float = 0.3,
        second_pair_thres: float | None = 0.7,
        nms_iou_thres: float = 0.5,
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
        dla_core: int | None = None,
        device: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        postprocessor: str | None = None,
        cuda_graph: bool | None = False,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        HandInteractionDetector.__init__(
            self,
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_range=input_range,
            preprocessor=preprocessor,
            resize_method=resize_method,
            conf_thres=conf_thres,
            pair_thres=pair_thres,
            second_pair_thres=second_pair_thres,
            nms_iou_thres=nms_iou_thres,
            mean=mean,
            std=std,
            dla_core=dla_core,
            device=device,
            backend=backend,
            warmup=warmup,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
            postprocessor=postprocessor,
            cuda_graph=cuda_graph,
            no_warn=no_warn,
            verbose=verbose,
        )
