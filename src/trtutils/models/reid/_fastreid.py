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


class FastReID(FeatureExtractor, Model):
    """
    Alias of FeatureExtractor with default args for FastReID person re-ID (JDAI-CV).

    The engine returns one 2048-d appearance embedding per person crop (the
    feature after the BN neck, as FastReID evaluates), L2 normalized by default
    so cosine similarity is a dot product. BoT and AGW models take
    ``(height, width) = (256, 128)`` crops, SBS models ``(384, 128)``; the
    exported size is read from the ONNX when building. FastReID normalizes
    0-255 RGB inside its network with the ImageNet mean / std scaled by 255,
    so the export starts after that step and the wrapper applies the same
    normalization on a 0-1 range. Engines are built with a dynamic batch (1 to
    ``batch_size``) so ``end2end_boxes`` can embed any number of detections.
    """

    _model_type = "fastreid"
    _friendly_name = "FastReID"
    _default_imgsz = (256, 128)
    _input_tensors: ClassVar[list[tuple[str, str]]] = [("input", "image")]
    _dynamic_batch = True

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
        normalize: bool = True,
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
