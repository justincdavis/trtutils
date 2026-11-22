# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from trtutils.image._feature_extractor import FeatureExtractor
from trtutils.models._hooks import vit_attention_fp32_build_hook
from trtutils.models._model import Model

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from typing_extensions import Self


class DINOv3(FeatureExtractor, Model):
    """
    Alias of FeatureExtractor with default args for DINOv3 ViT backbones.

    The engine returns the class token after the final norm as the embedding
    and the patch tokens as a (C, imgsz / 16, imgsz / 16) feature map; the
    register tokens are dropped. Any input size divisible by the patch size (16)
    can be downloaded, 256 is the timm default. The attention logits overflow
    FP16, so FP16 builds keep the attention matmul and softmax in FP32. The
    residual stream of ViT-L and ViT-H+ also exceeds the FP16 range, so the
    download scales it by 1/16 (the outputs are unchanged, every LayerNorm is
    scale invariant); build from ONNX made by ``download``.
    """

    _model_type = "dinov3"
    _friendly_name = "DINOv3"
    _default_imgsz = 256
    _imgsz_divisor = 16
    _input_tensors: ClassVar[list[tuple[str, str]]] = [("input", "image")]
    _build_hooks: ClassVar[list[Callable[..., dict[str, Any]]]] = [vit_attention_fp32_build_hook]

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
