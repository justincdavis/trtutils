# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from trtutils.image._image_restorer import ImageRestorer
from trtutils.models._model import Model

if TYPE_CHECKING:
    from pathlib import Path

    from typing_extensions import Self


class RealESRGAN(ImageRestorer, Model):
    """
    Alias of ImageRestorer with default args for Real-ESRGAN.

    The engine input size is static and images are never rescaled, so download and
    build at the size of your images (``imgsz=(height, width)``); smaller images are
    padded, larger ones are rejected. The default ``(480, 640)`` is only a starting
    point. ``realesrgan_x2plus`` needs an even height and width.
    """

    _model_type = "realesrgan"
    _friendly_name = "Real-ESRGAN"
    _default_imgsz = (480, 640)
    _input_tensors: ClassVar[list[tuple[str, str]]] = [("input", "image")]

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 1),
        preprocessor: str = "cuda",
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
        postprocessor: str | None = None,
        cpu_resize: bool | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        ImageRestorer.__init__(
            self,
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_range=input_range,
            preprocessor=preprocessor,
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
