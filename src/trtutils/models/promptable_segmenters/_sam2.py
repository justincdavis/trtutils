# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from trtutils.image._promptable_segmenter import ImageEncoder, MaskDecoder, PromptableSegmenter
from trtutils.models._model import Model

from ._sam import build_sam_decoder

if TYPE_CHECKING:
    from pathlib import Path

    from typing_extensions import Self

    from trtutils.compat._libs import cuda
    from trtutils.image._schema import (
        ImageEncoderOutputSchema,
        MaskDecoderInputSchema,
        MaskDecoderOutputSchema,
    )


class SAM2Encoder(ImageEncoder, Model):
    """Alias of ImageEncoder with default args for the SAM 2.1 (Hiera) image encoders."""

    _model_type = "sam2_encoder"
    _friendly_name = "SAM 2.1 encoder"
    _default_imgsz = 1024
    _valid_imgszs: ClassVar[list[int] | None] = [1024]
    _input_tensors: ClassVar[list[tuple[str, str]]] = [("images", "image")]
    # fp16 matches onnxruntime (cosine >= 0.99996); an fp16=False build runs in tf32, which
    # tensorrt enables by default, and its embeddings drop to cosine ~0.91
    _fp16 = True

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0.0, 1.0),
        preprocessor: str = "cuda",
        # SAM2ImagePredictor stretches the image to the input size
        resize_method: str = "linear",
        mean: tuple[float, float, float] | None = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] | None = (0.229, 0.224, 0.225),
        output_schema: ImageEncoderOutputSchema | str | None = None,
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
        ImageEncoder.__init__(
            self,
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_range=input_range,
            preprocessor=preprocessor,
            resize_method=resize_method,
            mean=mean,
            std=std,
            output_schema=output_schema,
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


class SAM2Decoder(MaskDecoder, Model):
    """Alias of MaskDecoder with default args for the SAM 2.1 (Hiera) mask decoders."""

    _model_type = "sam2_decoder"
    _friendly_name = "SAM 2.1 decoder"
    # the image size its point coordinates are in
    _default_imgsz = 1024
    _valid_imgszs: ClassVar[list[int] | None] = [1024]
    _input_tensors: ClassVar[list[tuple[str, Any]]] = [
        ("image_embeddings", (256, 64, 64)),
        ("high_res_feats_0", (32, 256, 256)),
        ("high_res_feats_1", (64, 128, 128)),
        ("point_coords", (1, 2)),
        ("point_labels", (1,)),
        ("mask_input", (1, 256, 256)),
        ("has_mask_input", ()),
    ]
    # fp16 saves little per call and costs mask accuracy
    _fp16 = False

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_schema: MaskDecoderInputSchema | str | None = None,
        output_schema: MaskDecoderOutputSchema | str | None = None,
        dla_core: int | None = None,
        device: int | None = None,
        backend: str = "auto",
        *,
        stream: cuda.cudaStream_t | None = None,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = None,
        shape_contexts: int | None = 4,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        MaskDecoder.__init__(
            self,
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_schema=input_schema,
            output_schema=output_schema,
            dla_core=dla_core,
            device=device,
            backend=backend,
            stream=stream,
            warmup=warmup,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
            cuda_graph=cuda_graph,
            shape_contexts=shape_contexts,
            no_warn=no_warn,
            verbose=verbose,
        )

    @classmethod
    def build(
        cls,
        onnx: Path | str,
        output: Path | str,
        imgsz: int | tuple[int, int] | None = None,
        batch_size: int = 4,
        dla_core: int | None = None,
        opt_level: int = 3,
        *,
        max_points: int = 8,
        fp16: bool = False,
        verbose: bool | None = None,
        embed_sst: bool = False,
        dynamic_batch: bool | None = None,  # noqa: ARG003
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """
        Build a TensorRT engine with a dynamic number of prompt sets and points.

        Parameters
        ----------
        onnx : Path | str
            Path to the ONNX model.
        output : Path | str
            Output path for the built engine.
        imgsz : int, optional
            The image encoder input size the point coordinates are in. Only 1024.
        batch_size : int
            The most prompt sets (e.g. boxes) per decoder call; more are run in
            chunks. Default is 4.
        dla_core : int | None
            The DLA core to build the engine for.
        opt_level : int
            TensorRT builder optimization level (0-5). Default is 3.
        max_points : int
            The most points per prompt set, counting the two corners of a box and
            the padding point every prompt gets. Default is 8.
        fp16 : bool
            Whether to build with FP16 precision enabled. Default is False: FP16
            saves little per call and costs mask accuracy.
        verbose : bool | None
            Enable verbose builder output.
        embed_sst : bool
            Not supported, the decoder has no image input; must be False.
        dynamic_batch : bool, optional
            Accepted for parity with ``Model.build`` and ignored: the number of
            prompt sets is always dynamic.
        **kwargs : Any
            Additional keyword arguments forwarded to ``build_engine()``.

        Raises
        ------
        ValueError
            If ``embed_sst`` is True or ``imgsz`` is not 1024.
        TypeError
            If ``imgsz`` is a (height, width) tuple.

        """
        if embed_sst:
            err_msg = "embed_sst is not supported for the SAM 2.1 decoder."
            raise ValueError(err_msg)
        size = imgsz if imgsz is not None else cls._default_imgsz
        if not isinstance(size, int):
            err_msg = f"{cls._friendly_name} takes a single imgsz, got {size}"
            raise TypeError(err_msg)
        cls._validate_imgsz(size)
        build_sam_decoder(
            onnx,
            output,
            batch_size,
            dla_core,
            opt_level,
            max_points=max_points,
            fp16=fp16,
            verbose=verbose,
            **kwargs,
        )


class SAM2(PromptableSegmenter):
    """Alias of PromptableSegmenter combining a SAM2Encoder and a SAM2Decoder."""

    def __init__(
        self: Self,
        encoder_engine: Path | str,
        decoder_engine: Path | str,
        mask_thres: float = 0.0,
        warmup_iterations: int = 10,
        preprocessor: str = "cuda",
        dla_core: int | None = None,
        device: int | None = None,
        backend: str = "auto",
        *,
        multimask: bool = True,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = None,
        cpu_resize: bool | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        """
        Create a SAM2 object from an image encoder and a mask decoder engine.

        The encoder and the decoder must come from the same checkpoint (e.g.
        ``sam2_1_hiera_tiny_encoder`` and ``sam2_1_hiera_tiny_decoder``). For custom encoder
        or decoder settings, create them directly and pass them to
        :class:`~trtutils.image.PromptableSegmenter`.

        Parameters
        ----------
        encoder_engine : Path, str
            The Path or str to the image encoder engine, see :meth:`SAM2Encoder.build`.
        decoder_engine : Path, str
            The Path or str to the mask decoder engine, see :meth:`SAM2Decoder.build`.
        mask_thres : float
            The logit above which a pixel is in the mask. By default 0.0.
        warmup_iterations : int
            The number of warmup iterations to perform.
            The default is 10.
        preprocessor : str
            The encoder's preprocessor, one of ['cpu', 'cuda', 'trt', 'dla0', 'dla1'].
            Default is 'cuda'.
        dla_core : int, optional
            The DLA core to assign DLA layers of the engines to. Default is None.
            If None, any DLA layers will be assigned to DLA core 0.
        device : int, optional
            The CUDA device index to use. Default is None,
            which uses the current device.
        backend : str
            The execution backend to use. Options are ['auto', 'async_v3', 'async_v2'].
            Default is 'auto', which selects the best available backend.
        multimask : bool
            Whether to pick the best of the decoder's multimask outputs instead of its
            single mask, see :class:`~trtutils.image.PromptableSegmenter`. By default True.
        warmup : bool, optional
            Whether or not to perform warmup iterations.
        pagelocked_mem : bool, optional
            Whether or not to use pagelocked memory for underlying CUDA operations.
            By default, pagelocked memory will be used.
        unified_mem : bool, optional
            Whether or not the system has unified memory.
            If True, use cudaHostAllocMapped to take advantage of unified memory.
            By default None, which means the default host allocation will be used.
        cuda_graph : bool, optional
            Whether or not to enable CUDA graph capture for both engines.
            Default is True.
        cpu_resize : bool, optional
            Whether the encoder resizes images on the CPU, see
            :class:`~trtutils.image.ImageModel`. Default is None.
        no_warn : bool, optional
            If True, suppresses warnings from TensorRT during engine deserialization.
            Default is None, which means warnings will be shown.
        verbose : bool, optional
            Whether or not to log additional information.
            Only covers the initialization phase.

        """
        encoder = SAM2Encoder(
            encoder_engine,
            warmup_iterations=warmup_iterations,
            preprocessor=preprocessor,
            dla_core=dla_core,
            device=device,
            backend=backend,
            warmup=warmup,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
            cuda_graph=cuda_graph,
            cpu_resize=cpu_resize,
            no_warn=no_warn,
            verbose=verbose,
        )
        # one stream: the decoder reads the embeddings right after the encoder writes them
        decoder = SAM2Decoder(
            decoder_engine,
            warmup_iterations=warmup_iterations,
            dla_core=dla_core,
            device=device,
            backend=backend,
            stream=encoder.engine.stream,
            warmup=warmup,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
            cuda_graph=cuda_graph,
            no_warn=no_warn,
            verbose=verbose,
        )
        PromptableSegmenter.__init__(
            self, encoder, decoder, mask_thres, multimask=multimask, verbose=verbose
        )
