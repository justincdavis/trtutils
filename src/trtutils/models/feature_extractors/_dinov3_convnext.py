# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from trtutils.image._feature_extractor import FeatureExtractor
from trtutils.models._model import Model

if TYPE_CHECKING:
    from pathlib import Path

    from typing_extensions import Self


class DINOv3ConvNeXt(FeatureExtractor, Model):
    """
    Alias of FeatureExtractor with default args for DINOv3 ConvNeXt backbones.

    The ConvNeXt distillations of DINOv3 (tiny, small, base, large). The engine
    returns the average-pooled last stage after the head norm as the embedding
    and the last stage itself as a (C, imgsz / 32, imgsz / 32) feature map. Any
    input size divisible by the total stride (32) can be downloaded, 224 is the
    timm default. FP16 builds need no layers pinned to FP32.
    """

    _model_type = "dinov3_convnext"
    _friendly_name = "DINOv3 ConvNeXt"
    _default_imgsz = 224
    _imgsz_divisor = 32
    _input_tensors: ClassVar[list[tuple[str, str]]] = [("input", "image")]

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 1),
        preprocessor: str = "cuda",
        resize_method: str = "linear",
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
        normalize: bool = False,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        FeatureExtractor.__init__(
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
            normalize=normalize,
            no_warn=no_warn,
            verbose=verbose,
        )
