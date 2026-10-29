# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from trtutils.builder._build import build_engine
from trtutils.image._ocr import OCR, TextDetector, TextRecognizer
from trtutils.models._model import Model

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self

    from trtutils.image._schema import TextDetectorOutputSchema, TextRecognizerOutputSchema

PPOCR_KEYS_V1 = Path(__file__).parent / "ppocr_keys_v1.txt"


class PPOCRDetector(TextDetector, Model):
    """Alias of TextDetector with default args for PP-OCR detection."""

    _model_type = "ppocr_det"
    _friendly_name = "PP-OCR detection"
    _default_imgsz = 960
    _input_tensors: ClassVar[list[tuple[str, str]]] = [("x", "image")]

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0.0, 1.0),
        preprocessor: str = "cpu",
        resize_method: str = "linear",
        conf_thres: float = 0.6,
        limit_side_len: int = 960,
        limit_type: str = "max",
        thresh: float = 0.3,
        max_candidates: int = 1000,
        unclip_ratio: float = 1.5,
        mean: tuple[float, float, float] | None = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] | None = (0.229, 0.224, 0.225),
        output_schema: TextDetectorOutputSchema | str | None = None,
        dla_core: int | None = None,
        device: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = None,
        postprocessor: str | None = None,
        use_dilation: bool | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        TextDetector.__init__(
            self,
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_range=input_range,
            preprocessor=preprocessor,
            resize_method=resize_method,
            conf_thres=conf_thres,
            limit_side_len=limit_side_len,
            limit_type=limit_type,
            thresh=thresh,
            max_candidates=max_candidates,
            unclip_ratio=unclip_ratio,
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
            use_dilation=use_dilation,
            no_warn=no_warn,
            verbose=verbose,
        )

    @classmethod
    def build(
        cls,
        onnx: Path | str,
        output: Path | str,
        imgsz: int | None = None,
        batch_size: int = 1,
        dla_core: int | None = None,
        opt_level: int = 3,
        *,
        min_imgsz: int = 32,
        max_imgsz: int = 2048,
        fp16: bool = True,
        verbose: bool | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """
        Build a TensorRT engine with a dynamic input height and width.

        Parameters
        ----------
        onnx : Path | str
            Path to the ONNX model.
        output : Path | str
            Output path for the built engine.
        imgsz : int, optional
            The height/width the engine is tuned for. Default is 960,
            the long side images are resized to.
        batch_size : int
            Maximum batch size of the engine; the profile starts at batch 1, which
            is what TextDetector submits. Default is 1.
        dla_core : int | None
            The DLA core to build the engine for.
        opt_level : int
            TensorRT builder optimization level (0-5). Default is 3.
        min_imgsz : int
            Smallest supported input height/width. Default is 32.
        max_imgsz : int
            Largest supported input height/width. Larger images are scaled down
            to fit at inference time. Default is 2048.
        fp16 : bool
            Whether to build with FP16 precision enabled. Default is True: on 24
            rendered pages FP16 changed ~4% of recognized lines with the same error
            rate against the ground truth (2.4%). Pass False to stay within float
            rounding of the FP32 reference (TensorRT may still use TF32).
        verbose : bool | None
            Enable verbose builder output.
        **kwargs : Any
            Additional keyword arguments forwarded to ``build_engine()``.

        """
        imgsz = imgsz if imgsz is not None else cls._default_imgsz
        build_engine(
            onnx=onnx,
            output=output,
            shapes=[
                (
                    cls._input_tensors[0][0],
                    (
                        (1, 3, min_imgsz, min_imgsz),
                        (1, 3, imgsz, imgsz),
                        (batch_size, 3, max_imgsz, max_imgsz),
                    ),
                ),
            ],
            fp16=fp16,
            dla_core=dla_core,
            optimization_level=opt_level,
            verbose=verbose,
            **kwargs,
        )


class PPOCRRecognizer(TextRecognizer, Model):
    """Alias of TextRecognizer with default args for PP-OCR recognition."""

    _model_type = "ppocr_rec"
    _friendly_name = "PP-OCR recognition"
    _default_imgsz = 48
    _input_tensors: ClassVar[list[tuple[str, str]]] = [("x", "image")]

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0.0, 1.0),
        preprocessor: str = "cpu",
        resize_method: str = "linear",
        character_dict: Path | str | Sequence[str] | None = PPOCR_KEYS_V1,
        mean: tuple[float, float, float] | None = (0.5, 0.5, 0.5),
        std: tuple[float, float, float] | None = (0.5, 0.5, 0.5),
        output_schema: TextRecognizerOutputSchema | str | None = None,
        dla_core: int | None = None,
        device: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = None,
        postprocessor: str | None = None,
        use_space_char: bool | None = None,
        width_bucket: int | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        TextRecognizer.__init__(
            self,
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_range=input_range,
            preprocessor=preprocessor,
            resize_method=resize_method,
            character_dict=character_dict,
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
            use_space_char=use_space_char,
            width_bucket=width_bucket,
            no_warn=no_warn,
            verbose=verbose,
        )

    @classmethod
    def build(
        cls,
        onnx: Path | str,
        output: Path | str,
        imgsz: int | None = None,
        batch_size: int = 6,
        dla_core: int | None = None,
        opt_level: int = 3,
        *,
        opt_width: int = 320,
        max_width: int = 3200,
        fp16: bool = True,
        verbose: bool | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """
        Build a TensorRT engine with a dynamic batch size and input width.

        Parameters
        ----------
        onnx : Path | str
            Path to the ONNX model.
        output : Path | str
            Output path for the built engine.
        imgsz : int, optional
            The input height of the model. Default is 48.
        batch_size : int
            Maximum number of text lines per inference. Default is 6.
        dla_core : int | None
            The DLA core to build the engine for.
        opt_level : int
            TensorRT builder optimization level (0-5). Default is 3.
        opt_width : int
            The input width the engine is tuned for. Default is 320.
        max_width : int
            Largest supported input width. Wider lines are squeezed to fit
            at inference time. Default is 3200.
        fp16 : bool
            Whether to build with FP16 precision enabled. Default is True: on 24
            rendered pages FP16 changed ~4% of recognized lines with the same error
            rate against the ground truth (2.4%). Pass False to stay within float
            rounding of the FP32 reference (TensorRT may still use TF32).
        verbose : bool | None
            Enable verbose builder output.
        **kwargs : Any
            Additional keyword arguments forwarded to ``build_engine()``.

        """
        height = imgsz if imgsz is not None else cls._default_imgsz
        build_engine(
            onnx=onnx,
            output=output,
            shapes=[
                (
                    cls._input_tensors[0][0],
                    (
                        (1, 3, height, 32),
                        (batch_size, 3, height, opt_width),
                        (batch_size, 3, height, max_width),
                    ),
                ),
            ],
            fp16=fp16,
            dla_core=dla_core,
            optimization_level=opt_level,
            verbose=verbose,
            **kwargs,
        )


class PPOCR(OCR):
    """Alias of OCR combining a PPOCRDetector and a PPOCRRecognizer."""

    def __init__(
        self: Self,
        det_engine: Path | str,
        rec_engine: Path | str,
        conf_thres: float = 0.5,
        min_height: int = 30,
        width_height_ratio: float = 8,
        max_side_len: int = 2000,
        min_side_len: int = 30,
        warmup_iterations: int = 10,
        character_dict: Path | str | Sequence[str] | None = PPOCR_KEYS_V1,
        dla_core: int | None = None,
        device: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        postprocessor: str | None = None,
        width_bucket: int | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        """
        Create a PPOCR object from a detection and a recognition engine.

        For custom detector or recognizer settings, create them directly and
        pass them to :class:`~trtutils.image.OCR`.

        Parameters
        ----------
        det_engine : Path, str
            The Path or str to the text detection engine, see :meth:`PPOCRDetector.build`.
        rec_engine : Path, str
            The Path or str to the text recognition engine, see :meth:`PPOCRRecognizer.build`.
        conf_thres : float
            The minimum recognition confidence for a line to be returned.
            By default 0.5.
        min_height : int
            Images at most this tall are padded vertically before detection.
            By default 30.
        width_height_ratio : float
            Images wider than this aspect ratio are padded vertically before
            detection. Set to -1 to disable. By default 8.
        max_side_len : int
            Images with a longer side above this are downscaled first.
            By default 2000.
        min_side_len : int
            Images with a shorter side below this are upscaled first.
            By default 30.
        warmup_iterations : int
            The number of warmup iterations to perform.
            The default is 10.
        character_dict : Path, str, Sequence[str]
            The recognizer's character dictionary. By default ``ppocr_keys_v1.txt``.
        dla_core : int, optional
            The DLA core to assign DLA layers of the engines to. Default is None.
            If None, any DLA layers will be assigned to DLA core 0.
        device : int, optional
            The CUDA device index to use. Default is None,
            which uses the current device.
        backend : str
            The execution backend to use. Options are ['auto', 'async_v3', 'async_v2'].
            Default is 'auto', which selects the best available backend.
        warmup : bool, optional
            Whether or not to perform warmup iterations.
        pagelocked_mem : bool, optional
            Whether or not to use pagelocked memory for underlying CUDA operations.
            By default, pagelocked memory will be used.
        unified_mem : bool, optional
            Whether or not the system has unified memory.
            If True, use cudaHostAllocMapped to take advantage of unified memory.
            By default None, which means the default host allocation will be used.
        postprocessor : str, optional
            Where to pre- and postprocess, 'cpu' or 'cuda', for both models. 'cuda' uploads
            the image once and runs detector preprocessing, text line crops, recognizer
            preprocessing and the decodes on the GPU with results identical to the CPU
            path. By default None, which uses 'cuda'.
        width_bucket : int, optional
            Round the recognizer's batch widths up to a multiple of this many pixels and keep
            one execution context per width, see :class:`~trtutils.image.TextRecognizer`.
            Faster steady state, but each width costs ~65 ms on first use and the extra
            padding changes outputs slightly. By default None.
        no_warn : bool, optional
            If True, suppresses warnings from TensorRT during engine deserialization.
            Default is None, which means warnings will be shown.
        verbose : bool, optional
            Whether or not to log additional information.
            Only covers the initialization phase.

        """
        OCR.__init__(
            self,
            PPOCRDetector(
                det_engine,
                warmup_iterations=warmup_iterations,
                dla_core=dla_core,
                device=device,
                backend=backend,
                warmup=warmup,
                pagelocked_mem=pagelocked_mem,
                unified_mem=unified_mem,
                postprocessor=postprocessor,
                no_warn=no_warn,
                verbose=verbose,
            ),
            PPOCRRecognizer(
                rec_engine,
                character_dict=character_dict,
                warmup_iterations=warmup_iterations,
                dla_core=dla_core,
                device=device,
                backend=backend,
                warmup=warmup,
                pagelocked_mem=pagelocked_mem,
                unified_mem=unified_mem,
                postprocessor=postprocessor,
                width_bucket=width_bucket,
                no_warn=no_warn,
                verbose=verbose,
            ),
            conf_thres,
            min_height,
            width_height_ratio,
            max_side_len,
            min_side_len,
            verbose=verbose,
        )
