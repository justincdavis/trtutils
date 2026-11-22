# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from trtutils.image._semantic_segmenter import SemanticSegmenter
from trtutils.models._hooks import argmax_build_hook
from trtutils.models._model import Model

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from typing_extensions import Self


class DeepLabV3(SemanticSegmenter, Model):
    """
    Alias of SemanticSegmenter with default args for torchvision DeepLabV3.

    The default weights predict the 21 Pascal VOC classes (0 is background).
    ``build()`` runs the argmax in the engine; pass ``argmax=False`` to keep the logits.
    """

    _model_type = "deeplabv3"
    _friendly_name = "DeepLabV3"
    _default_imgsz = 520
    _input_tensors: ClassVar[list[tuple[str, str]]] = [("input", "image")]
    _build_hooks: ClassVar[list[Callable[..., dict[str, Any]]]] = [argmax_build_hook]

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 1),
        preprocessor: str = "cuda",
        resize_method: str = "letterbox",
        mean: tuple[float, float, float] | None = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] | None = (0.229, 0.224, 0.225),
        dla_core: int | None = None,
        device: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = None,
        postprocessor: str | None = None,
        cpu_resize: bool | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        SemanticSegmenter.__init__(
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
            postprocessor=postprocessor,
            cpu_resize=cpu_resize,
            no_warn=no_warn,
            verbose=verbose,
        )
