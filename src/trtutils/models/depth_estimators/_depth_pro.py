# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from trtutils.image._depth_estimator import DepthEstimator
from trtutils.models._model import Model

if TYPE_CHECKING:
    from pathlib import Path

    from typing_extensions import Self


class DepthPro2(DepthEstimator, Model):
    """
    Alias of DepthEstimator with default args for Apple's Depth Pro 2 (image mode).

    The engine outputs metric depth in metres along the optical axis, using the focal
    length that Depth Pro 2's own estimator predicts from the image, so ``normalize``
    defaults to False. Depth is infinite where the network sees no finite distance, such
    as the sky. The maps have the engine's input size; resize them to the image size for
    per-pixel metres. The estimated focal length, in pixels of the engine input, is the
    second output of each image from :meth:`run`: ``model.run(image)[1]`` for one image,
    ``model.run(images)[i][1]`` for a list.

    Any ``(height, width)`` divisible by 32 can be exported; match the camera's aspect
    ratio, as the image is resized without padding.
    The weights are licensed for non-commercial research use only.
    """

    _model_type = "depth_pro_2"
    _friendly_name = "Depth-Pro-2"
    _default_imgsz = (576, 1024)
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
        DepthEstimator.__init__(
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
