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


class DeiT(Classifier, Model):
    """
    Alias of Classifier with default args for DeiT.

    Any input size divisible by the patch size (16) can be downloaded: the
    position embeddings are resampled to it. ``*_hr`` models are fine-tuned
    at 384 and download at 384 by default. Their reference crop_pct is 1.0,
    the others use 0.9.
    """

    _model_type = "deit"
    _friendly_name = "DeiT"
    _default_imgsz = 224
    _input_tensors: ClassVar[list[tuple[str, str]]] = [("input", "image")]

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 1),
        preprocessor: str = "trt",
        resize_method: str = "center_crop",
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
        crop_pct: float = 0.9,
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
            postprocessor=postprocessor,
            crop_pct=crop_pct,
            no_warn=no_warn,
            verbose=verbose,
        )


class DeiT3(DeiT):
    """
    Alias of Classifier with default args for DeiT III.

    Same defaults and download rules as DeiT; ``deit3_huge`` uses a patch
    size of 14. ``*_in22k`` models are pretrained on ImageNet-22k and
    fine-tuned on ImageNet-1k. The reference crop_pct is 0.9 for the
    ImageNet-1k models at 224 and 1.0 for the others.
    """

    _model_type = "deit3"
    _friendly_name = "DeiT III"
