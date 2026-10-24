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

PPOCR_KEYS_V1 = Path(__file__).parent / "ppocr_keys_v1.txt"


class PPOCRDetector(TextDetector, Model):
    """
    Alias of TextDetector for PaddleOCR (PP-OCR) DB text detection models.

    Download with ``PPOCRDetector.download("ppocrv4_det", ...)`` and build with
    :meth:`build`, which creates a dynamic-shape engine.
    """

    _model_type = "ppocr_det"
    _friendly_name = "PP-OCR detection"
    _default_imgsz = 736
    _input_tensors: ClassVar[list[tuple[str, str]]] = [("x", "image")]

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
        fp16: bool = False,
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
            The height/width the engine is tuned for. Default is 736.
        batch_size : int
            Batch size of the engine. Default is 1.
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
            Whether to build with FP16 precision enabled. Default is False, which
            reproduces the reference FP32 outputs exactly; FP16 can change a
            small fraction of recognized lines.
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
                        (batch_size, 3, min_imgsz, min_imgsz),
                        (batch_size, 3, imgsz, imgsz),
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
    """
    Alias of TextRecognizer for PaddleOCR (PP-OCR) CTC text recognition models.

    Defaults to the PP-OCR Chinese/English character dictionary (``ppocr_keys_v1.txt``).
    Download with ``PPOCRRecognizer.download("ppocrv4_rec", ...)`` and build with
    :meth:`build`, which creates a dynamic-shape engine.
    """

    _model_type = "ppocr_rec"
    _friendly_name = "PP-OCR recognition"
    _default_imgsz = 48
    _input_tensors: ClassVar[list[tuple[str, str]]] = [("x", "image")]

    def __init__(
        self: Self,
        engine_path: Path | str,
        character_dict: Path | str | Sequence[str] = PPOCR_KEYS_V1,
        warmup_iterations: int = 1,
        backend: str = "auto",
        device: int | None = None,
        *,
        use_space_char: bool = True,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        TextRecognizer.__init__(
            self,
            engine_path=engine_path,
            character_dict=character_dict,
            warmup_iterations=warmup_iterations,
            backend=backend,
            device=device,
            use_space_char=use_space_char,
            warmup=warmup,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
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
        fp16: bool = False,
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
            Whether to build with FP16 precision enabled. Default is False, which
            reproduces the reference FP32 outputs exactly; FP16 can change a
            small fraction of recognized lines.
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
        character_dict: Path | str | Sequence[str] = PPOCR_KEYS_V1,
        text_score: float = 0.5,
        device: int | None = None,
        *,
        warmup: bool | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        """
        Create a PP-OCR pipeline from a detection and a recognition engine.

        Parameters
        ----------
        det_engine : Path | str
            Path to the text detection engine (see :meth:`PPOCRDetector.build`).
        rec_engine : Path | str
            Path to the text recognition engine (see :meth:`PPOCRRecognizer.build`).
        character_dict : Path | str | Sequence[str]
            The recognizer's character dictionary. Default is ``ppocr_keys_v1.txt``.
        text_score : float
            Minimum recognition confidence for a line to be returned. Default is 0.5.
        device : int, optional
            The CUDA device index to use. Default is None (current device).
        warmup : bool, optional
            Whether or not to perform warmup iterations.
        no_warn : bool, optional
            Suppress TensorRT warnings during engine deserialization.
        verbose : bool, optional
            Whether or not to log additional information.

        """
        OCR.__init__(
            self,
            detector=PPOCRDetector(
                det_engine, device=device, warmup=warmup, no_warn=no_warn, verbose=verbose
            ),
            recognizer=PPOCRRecognizer(
                rec_engine,
                character_dict,
                device=device,
                warmup=warmup,
                no_warn=no_warn,
                verbose=verbose,
            ),
            text_score=text_score,
        )
