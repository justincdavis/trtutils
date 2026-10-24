# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
PaddleOCR (PP-OCR) style text detection, text recognition and two-stage OCR.

Classes
-------
:class:`TextDetector`
    DB text detector on a dynamic-shape engine.
:class:`TextRecognizer`
    CTC text line recognizer on a dynamic-shape engine.
:class:`OCR`
    Detection followed by recognition of every detected line.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from trtutils._engine import TRTEngine
from trtutils._log import LOG
from trtutils.core._buffer import Buffer

from .postprocessors._ocr import postprocess_text_detection, postprocess_text_recognition
from .preprocessors._ocr import (
    _IMAGENET_MEAN,
    _IMAGENET_STD,
    crop_text_region,
    preprocess_text_detection,
    preprocess_text_recognition,
    resize_text_detection,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self


def _make_engine(
    engine_path: Path | str,
    warmup_iterations: int,
    backend: str,
    device: int | None,
    *,
    dla_core: int | None,
    warmup: bool | None,
    pagelocked_mem: bool | None,
    unified_mem: bool | None,
    no_warn: bool | None,
    verbose: bool | None,
) -> tuple[TRTEngine, tuple[int, ...]]:
    """Load an engine and return it with the max shape of its (dynamic) input profile."""
    engine = TRTEngine(
        engine_path=engine_path,
        warmup_iterations=warmup_iterations,
        backend=backend,
        dla_core=dla_core,
        device=device,
        warmup=warmup,
        pagelocked_mem=pagelocked_mem,
        unified_mem=unified_mem,
        # input shapes change on every call, so a captured graph would never be reused
        cuda_graph=False,
        no_warn=no_warn,
        verbose=verbose,
    )
    return engine, engine.input_bindings[0].max_shape


def _host_image(image: np.ndarray | Buffer) -> np.ndarray:
    return image.numpy() if isinstance(image, Buffer) else image


class TextDetector:
    """
    DB (Differentiable Binarization) text detector, as used by PaddleOCR (PP-OCR).

    The engine should be built with a dynamic (min, opt, max) profile over the
    input height and width, see :meth:`trtutils.models.PPOCRDetector.build`.
    CUDA graphs are not used since the input shape changes on almost every call.
    """

    def __init__(
        self: Self,
        engine_path: Path | str,
        limit_side_len: int = 960,
        limit_type: str = "max",
        thresh: float = 0.3,
        box_thresh: float = 0.6,
        max_candidates: int = 1000,
        unclip_ratio: float = 1.5,
        mean: tuple[float, float, float] = _IMAGENET_MEAN,
        std: tuple[float, float, float] = _IMAGENET_STD,
        warmup_iterations: int = 1,
        backend: str = "auto",
        device: int | None = None,
        *,
        dla_core: int | None = None,
        use_dilation: bool = False,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        """
        Create a TextDetector.

        The defaults are PaddleOCR's (``tools/infer/utility.py``): longer side
        limited to 960, ImageNet normalization, box threshold 0.6, unclip 1.5,
        no dilation. RapidOCR instead uses ``limit_side_len=736, limit_type="min"``,
        mean/std 0.5, ``box_thresh=0.5``, ``unclip_ratio=1.6`` and dilation.

        Parameters
        ----------
        engine_path : Path | str
            The path to the compiled TensorRT engine.
        limit_side_len : int
            Images are resized so their shorter (``limit_type="min"``) or longer
            (``limit_type="max"``) side matches this length. Default is 960.
        limit_type : str
            ``"min"`` or ``"max"``. Default is ``"max"``.
        thresh : float
            Binarization threshold of the probability map. Default is 0.3.
        box_thresh : float
            Minimum mean probability inside a box to keep it. Default is 0.6.
        max_candidates : int
            Maximum number of candidate regions. Default is 1000.
        unclip_ratio : float
            How far to expand detected text kernels. Default is 1.5.
        mean : tuple[float, float, float]
            Normalization mean. Default is the ImageNet mean (0.485, 0.456, 0.406).
        std : tuple[float, float, float]
            Normalization standard deviation. Default is the ImageNet std
            (0.229, 0.224, 0.225).
        warmup_iterations : int
            The number of warmup iterations to perform. Default is 1.
        backend : str
            The execution backend to use. Default is 'auto'.
        device : int, optional
            The CUDA device index to use. Default is None (current device).
        dla_core : int, optional
            The DLA core the engine was built for. Default is None.
        use_dilation : bool
            Whether to dilate the binarized map before extracting boxes. Default is False.
        warmup : bool, optional
            Whether or not to perform warmup iterations.
        pagelocked_mem : bool, optional
            Whether or not to use pagelocked memory.
        unified_mem : bool, optional
            Whether or not the system has unified memory.
        no_warn : bool, optional
            Suppress TensorRT warnings during engine deserialization.
        verbose : bool, optional
            Whether or not to log additional information.

        """
        self._engine, max_shape = _make_engine(
            engine_path,
            warmup_iterations,
            backend,
            device,
            dla_core=dla_core,
            warmup=warmup,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
            no_warn=no_warn,
            verbose=verbose,
        )
        self._max_hw: tuple[int, int] = (max_shape[2], max_shape[3])
        self._limit_side_len = limit_side_len
        self._limit_type = limit_type
        self._mean = mean
        self._std = std
        self._thresh = thresh
        self._box_thresh = box_thresh
        self._max_candidates = max_candidates
        self._unclip_ratio = unclip_ratio
        self._use_dilation = use_dilation
        self._verbose = verbose if verbose is not None else False
        if self._verbose:
            LOG.debug(f"Created TextDetector: {self.name}, max input {self._max_hw}")

    @property
    def engine(self: Self) -> TRTEngine:
        """Get the underlying TRTEngine."""
        return self._engine

    @property
    def name(self: Self) -> str:
        """Get the name of the engine."""
        return self._engine.name

    def preprocess(self: Self, image: np.ndarray | Buffer) -> np.ndarray:
        """
        Preprocess an image for the detector.

        Parameters
        ----------
        image : np.ndarray | Buffer
            The image in HWC, BGR, uint8 format.

        Returns
        -------
        np.ndarray
            The input tensor of shape (1, 3, H, W).

        """
        return preprocess_text_detection(
            _host_image(image),
            self._limit_side_len,
            self._limit_type,
            self._mean,
            self._std,
            self._max_hw,
        )

    def postprocess(
        self: Self,
        output: np.ndarray,
        image_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract text boxes from the detector output.

        Parameters
        ----------
        output : np.ndarray
            The probability map output by the engine.
        image_shape : tuple[int, int]
            The (height, width) of the original image.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The boxes (N, 4, 2) in reading order and their scores (N,).

        """
        return postprocess_text_detection(
            output,
            image_shape,
            self._thresh,
            self._box_thresh,
            self._max_candidates,
            self._unclip_ratio,
            use_dilation=self._use_dilation,
        )

    def run(self: Self, image: np.ndarray | Buffer) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect text in an image.

        Parameters
        ----------
        image : np.ndarray | Buffer
            The image in HWC, BGR, uint8 format.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The boxes, shape (N, 4, 2) float32 corner points (TL, TR, BR, BL)
            in image coordinates and reading order, and their scores, shape (N,).

        """
        tensor = self.preprocess(image)
        output = self._engine.execute([Buffer.wrap(tensor)])[0]
        h, w = image.shape[:2]
        return self.postprocess(output, (h, w))

    def __call__(self: Self, image: np.ndarray | Buffer) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect text in an image, see :meth:`run`.

        Parameters
        ----------
        image : np.ndarray | Buffer
            The image in HWC, BGR, uint8 format.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The boxes and their scores.

        """
        return self.run(image)


class TextRecognizer:
    """
    CTC text line recognizer, as used by PaddleOCR (PP-OCR).

    The engine should be built with a dynamic (min, opt, max) profile over the
    batch size and input width, see :meth:`trtutils.models.PPOCRRecognizer.build`.
    CUDA graphs are not used since the input shape changes on almost every call.
    """

    def __init__(
        self: Self,
        engine_path: Path | str,
        character_dict: Path | str | Sequence[str],
        warmup_iterations: int = 1,
        backend: str = "auto",
        device: int | None = None,
        *,
        dla_core: int | None = None,
        use_space_char: bool = True,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        """
        Create a TextRecognizer.

        Parameters
        ----------
        engine_path : Path | str
            The path to the compiled TensorRT engine.
        character_dict : Path | str | Sequence[str]
            The model's character dictionary: a text file with one character
            per line, or the list of characters itself (excluding the CTC blank).
        warmup_iterations : int
            The number of warmup iterations to perform. Default is 1.
        backend : str
            The execution backend to use. Default is 'auto'.
        device : int, optional
            The CUDA device index to use. Default is None (current device).
        dla_core : int, optional
            The DLA core the engine was built for. Default is None.
        use_space_char : bool
            Whether the model has an extra trailing class for the space character.
            Default is True.
        warmup : bool, optional
            Whether or not to perform warmup iterations.
        pagelocked_mem : bool, optional
            Whether or not to use pagelocked memory.
        unified_mem : bool, optional
            Whether or not the system has unified memory.
        no_warn : bool, optional
            Suppress TensorRT warnings during engine deserialization.
        verbose : bool, optional
            Whether or not to log additional information.

        Raises
        ------
        ValueError
            If the character dictionary does not match the engine's number of classes.

        """
        self._engine, max_shape = _make_engine(
            engine_path,
            warmup_iterations,
            backend,
            device,
            dla_core=dla_core,
            warmup=warmup,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
            no_warn=no_warn,
            verbose=verbose,
        )
        self._batch_size: int = max_shape[0]
        self._image_height: int = max_shape[2]
        self._max_width: int = max_shape[3]

        if isinstance(character_dict, (str, Path)):
            with Path(character_dict).open(encoding="utf-8") as f:
                chars = [line.rstrip("\r\n") for line in f]
            # a trailing newline is not a character
            if chars and not chars[-1]:
                chars.pop()
        else:
            chars = list(character_dict)
        self._characters: list[str] = ["blank", *chars] + ([" "] if use_space_char else [])

        num_classes = self._engine.output_spec[0][0][-1]
        if num_classes > 0 and num_classes != len(self._characters):
            err_msg = (
                f"Character dictionary has {len(self._characters)} classes (with blank"
                f"{' and space' if use_space_char else ''}), engine outputs {num_classes}"
            )
            raise ValueError(err_msg)
        self._verbose = verbose if verbose is not None else False
        if self._verbose:
            LOG.debug(f"Created TextRecognizer: {self.name}, max input {max_shape}")

    @property
    def engine(self: Self) -> TRTEngine:
        """Get the underlying TRTEngine."""
        return self._engine

    @property
    def name(self: Self) -> str:
        """Get the name of the engine."""
        return self._engine.name

    @property
    def characters(self: Self) -> list[str]:
        """Get the class labels, index 0 is the CTC blank."""
        return self._characters

    def preprocess(self: Self, images: Sequence[np.ndarray | Buffer]) -> np.ndarray:
        """
        Preprocess a batch of text line images.

        Parameters
        ----------
        images : Sequence[np.ndarray | Buffer]
            Text line images in HWC, BGR, uint8 format, at most the engine's
            max batch size.

        Returns
        -------
        np.ndarray
            The input tensor of shape (N, 3, H, W).

        """
        return preprocess_text_recognition(
            [_host_image(img) for img in images], self._image_height, max_width=self._max_width
        )

    def postprocess(self: Self, output: np.ndarray) -> list[tuple[str, float]]:
        """
        Decode the recognizer output.

        Parameters
        ----------
        output : np.ndarray
            The per-timestep class probabilities, shape (N, T, C).

        Returns
        -------
        list[tuple[str, float]]
            The text and confidence of each line.

        """
        return postprocess_text_recognition(output, self._characters)

    def run(
        self: Self,
        images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
    ) -> list[tuple[str, float]]:
        """
        Recognize the text in one or more text line images.

        Lines are grouped by aspect ratio into batches of the engine's max
        batch size, so similar widths share padding.

        Parameters
        ----------
        images : np.ndarray | Buffer | Sequence[np.ndarray | Buffer]
            A text line image, or a list of them, in HWC, BGR, uint8 format.

        Returns
        -------
        list[tuple[str, float]]
            The text and confidence of each line, in input order.

        """
        lines = [images] if isinstance(images, (np.ndarray, Buffer)) else images
        order = np.argsort([img.shape[1] / img.shape[0] for img in lines])
        results: list[tuple[str, float]] = [("", 0.0)] * len(lines)
        for start in range(0, len(order), self._batch_size):
            batch_idx = order[start : start + self._batch_size]
            tensor = self.preprocess([lines[i] for i in batch_idx])
            output = self._engine.execute([Buffer.wrap(tensor)])[0]
            for i, res in zip(batch_idx, self.postprocess(output)):
                results[i] = res
        return results

    def __call__(
        self: Self,
        images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
    ) -> list[tuple[str, float]]:
        """
        Recognize text line images, see :meth:`run`.

        Parameters
        ----------
        images : np.ndarray | Buffer | Sequence[np.ndarray | Buffer]
            A text line image, or a list of them.

        Returns
        -------
        list[tuple[str, float]]
            The text and confidence of each line.

        """
        return self.run(images)


class OCR:
    """
    Two-stage OCR pipeline: text detection followed by text line recognition.

    Detected regions are cropped (perspective-corrected), recognized in
    batches, and low-confidence lines are dropped. Before detection, very
    large, small or thin images are resized and padded the way RapidOCR does.
    """

    def __init__(
        self: Self,
        detector: TextDetector,
        recognizer: TextRecognizer,
        text_score: float = 0.5,
        min_height: int = 30,
        width_height_ratio: float = 8,
        max_side_len: int = 2000,
        min_side_len: int = 30,
    ) -> None:
        """
        Create an OCR pipeline.

        Parameters
        ----------
        detector : TextDetector
            The text detector.
        recognizer : TextRecognizer
            The text recognizer.
        text_score : float
            Minimum recognition confidence for a line to be returned. Default is 0.5.
        min_height : int
            Images at most this tall are padded vertically before detection,
            which helps the detector on single text lines. Default is 30.
        width_height_ratio : float
            Images wider than this aspect ratio are padded vertically before
            detection. Set to -1 to disable. Default is 8.
        max_side_len : int
            Images with a longer side above this are downscaled first. Default is 2000.
        min_side_len : int
            Images with a shorter side below this are upscaled first. Default is 30.

        """
        self._detector = detector
        self._recognizer = recognizer
        self._text_score = text_score
        self._min_height = min_height
        self._width_height_ratio = width_height_ratio
        self._max_side_len = max_side_len
        self._min_side_len = min_side_len

    @property
    def detector(self: Self) -> TextDetector:
        """Get the text detector."""
        return self._detector

    @property
    def recognizer(self: Self) -> TextRecognizer:
        """Get the text recognizer."""
        return self._recognizer

    def run(self: Self, image: np.ndarray | Buffer) -> list[tuple[np.ndarray, str, float]]:
        """
        Detect and recognize all text in an image.

        Parameters
        ----------
        image : np.ndarray | Buffer
            The image in HWC, BGR, uint8 format.

        Returns
        -------
        list[tuple[np.ndarray, str, float]]
            ``(box, text, score)`` per text line in reading order, where ``box``
            is the (4, 2) float32 corner points (TL, TR, BR, BL) in the
            coordinates of ``image``.

        """
        img = _host_image(image)
        raw_h, raw_w = img.shape[:2]
        if max(raw_h, raw_w) > self._max_side_len:
            img = resize_text_detection(img, self._max_side_len, "max")
        if min(img.shape[:2]) < self._min_side_len:
            img = resize_text_detection(img, self._min_side_len, "min")

        h, w = img.shape[:2]
        scale = np.array([raw_w / w, raw_h / h], np.float32)
        pad = 0
        if h <= self._min_height or (
            self._width_height_ratio != -1 and w / h > self._width_height_ratio
        ):
            new_h = max(int(w / self._width_height_ratio), self._min_height) * 2
            pad = int(abs(new_h - h) / 2)
            img = cv2.copyMakeBorder(img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        boxes, _ = self._detector.run(img)
        if len(boxes) == 0:
            return []
        lines = self._recognizer.run([crop_text_region(img, box) for box in boxes])

        out_boxes = (boxes - np.array([0, pad], np.float32)) * scale
        out_boxes = np.clip(out_boxes, 0, np.array([raw_w, raw_h], np.float32))
        return [
            (box, text, score)
            for box, (text, score) in zip(out_boxes, lines)
            if score >= self._text_score
        ]

    def __call__(self: Self, image: np.ndarray | Buffer) -> list[tuple[np.ndarray, str, float]]:
        """
        Detect and recognize all text in an image, see :meth:`run`.

        Parameters
        ----------
        image : np.ndarray | Buffer
            The image in HWC, BGR, uint8 format.

        Returns
        -------
        list[tuple[np.ndarray, str, float]]
            ``(box, text, score)`` per text line.

        """
        return self.run(image)
