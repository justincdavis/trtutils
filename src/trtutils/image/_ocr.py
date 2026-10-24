# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
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
    crop_text_region,
    preprocess_text_detection,
    preprocess_text_recognition,
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
        device=device,
        warmup=warmup,
        pagelocked_mem=pagelocked_mem,
        unified_mem=unified_mem,
        # input shapes change on every call, so a captured graph would never be reused
        cuda_graph=False,
        no_warn=no_warn,
        verbose=verbose,
    )
    max_shape = tuple(engine.engine.get_tensor_profile_shape(engine.input_names[0], 0)[2])
    return engine, max_shape


class TextDetector:
    """
    DB (Differentiable Binarization) text detector, as used by PaddleOCR (PP-OCR).

    The engine should be built with a dynamic (min, opt, max) profile over the
    input height and width, see :meth:`trtutils.models.PPOCRDetector.build`.
    """

    def __init__(
        self: Self,
        engine_path: Path | str,
        limit_side_len: int = 736,
        limit_type: str = "min",
        thresh: float = 0.3,
        box_thresh: float = 0.5,
        max_candidates: int = 1000,
        unclip_ratio: float = 1.6,
        mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: tuple[float, float, float] = (0.5, 0.5, 0.5),
        warmup_iterations: int = 1,
        backend: str = "auto",
        device: int | None = None,
        *,
        use_dilation: bool = True,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        """
        Create a TextDetector.

        Parameters
        ----------
        engine_path : Path | str
            The path to the compiled TensorRT engine.
        limit_side_len : int
            Images are resized so their shorter (``limit_type="min"``) or longer
            (``limit_type="max"``) side matches this length. Default is 736.
        limit_type : str
            ``"min"`` or ``"max"``. Default is ``"min"``.
        thresh : float
            Binarization threshold of the probability map. Default is 0.3.
        box_thresh : float
            Minimum mean probability inside a box to keep it. Default is 0.5.
        max_candidates : int
            Maximum number of candidate regions. Default is 1000.
        unclip_ratio : float
            How far to expand detected text kernels. Default is 1.6.
        mean : tuple[float, float, float]
            Normalization mean. Default is (0.5, 0.5, 0.5).
        std : tuple[float, float, float]
            Normalization standard deviation. Default is (0.5, 0.5, 0.5).
        warmup_iterations : int
            The number of warmup iterations to perform. Default is 1.
        backend : str
            The execution backend to use. Default is 'auto'.
        device : int, optional
            The CUDA device index to use. Default is None (current device).
        use_dilation : bool
            Whether to dilate the binarized map before extracting boxes. Default is True.
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
        LOG.debug(f"Created TextDetector: {self.name}, max input {self._max_hw}")

    @property
    def engine(self: Self) -> TRTEngine:
        """Get the underlying TRTEngine."""
        return self._engine

    @property
    def name(self: Self) -> str:
        """Get the name of the engine."""
        return self._engine.name

    def preprocess(self: Self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess an image for the detector.

        Parameters
        ----------
        image : np.ndarray
            The image in HWC, BGR, uint8 format.

        Returns
        -------
        np.ndarray
            The input tensor of shape (1, 3, H, W).

        """
        return preprocess_text_detection(
            image,
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

    def run(self: Self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect text in an image.

        Parameters
        ----------
        image : np.ndarray
            The image in HWC, BGR, uint8 format.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The boxes, shape (N, 4, 2) float32 corner points (TL, TR, BR, BL)
            in image coordinates and reading order, and their scores, shape (N,).

        """
        tensor = self.preprocess(image)
        output = self._engine.execute([Buffer.wrap(tensor)])[0]
        return self.postprocess(output, image.shape[:2])

    def __call__(self: Self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect text in an image, see :meth:`run`.

        Parameters
        ----------
        image : np.ndarray
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
    """

    def __init__(
        self: Self,
        engine_path: Path | str,
        character_dict: Path | str | Sequence[str],
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

    def preprocess(self: Self, images: Sequence[np.ndarray]) -> np.ndarray:
        """
        Preprocess a batch of text line images.

        Parameters
        ----------
        images : Sequence[np.ndarray]
            Text line images in HWC, BGR, uint8 format, at most the engine's
            max batch size.

        Returns
        -------
        np.ndarray
            The input tensor of shape (N, 3, H, W).

        """
        return preprocess_text_recognition(images, self._image_height, max_width=self._max_width)

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
        images: np.ndarray | Sequence[np.ndarray],
    ) -> list[tuple[str, float]]:
        """
        Recognize the text in one or more text line images.

        Lines are grouped by aspect ratio into batches of the engine's max
        batch size, so similar widths share padding.

        Parameters
        ----------
        images : np.ndarray | Sequence[np.ndarray]
            A text line image, or a list of them, in HWC, BGR, uint8 format.

        Returns
        -------
        list[tuple[str, float]]
            The text and confidence of each line, in input order.

        """
        if isinstance(images, np.ndarray):
            images = [images]
        order = np.argsort([img.shape[1] / img.shape[0] for img in images])
        results: list[tuple[str, float]] = [("", 0.0)] * len(images)
        for start in range(0, len(order), self._batch_size):
            batch_idx = order[start : start + self._batch_size]
            tensor = self.preprocess([images[i] for i in batch_idx])
            output = self._engine.execute([Buffer.wrap(tensor)])[0]
            for i, res in zip(batch_idx, self.postprocess(output)):
                results[i] = res
        return results

    def __call__(
        self: Self,
        images: np.ndarray | Sequence[np.ndarray],
    ) -> list[tuple[str, float]]:
        """
        Recognize text line images, see :meth:`run`.

        Parameters
        ----------
        images : np.ndarray | Sequence[np.ndarray]
            A text line image, or a list of them.

        Returns
        -------
        list[tuple[str, float]]
            The text and confidence of each line.

        """
        return self.run(images)


def _fit_side(image: np.ndarray, side: int, *, shrink: bool) -> tuple[np.ndarray, float, float]:
    """Resize so the longer side is <= side (shrink) or the shorter side >= side, to multiples of 32."""
    h, w = image.shape[:2]
    if (shrink and max(h, w) <= side) or (not shrink and min(h, w) >= side):
        return image, 1.0, 1.0
    ratio = side / (max(h, w) if shrink else min(h, w))
    new_h = max(32, int(round(int(h * ratio) / 32) * 32))
    new_w = max(32, int(round(int(w * ratio) / 32) * 32))
    return cv2.resize(image, (new_w, new_h)), h / new_h, w / new_w


class OCR:
    """
    Two-stage OCR pipeline: text detection followed by text line recognition.

    Detected regions are cropped (perspective-corrected), recognized in
    batches, and low-confidence lines are dropped.
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

    def run(self: Self, image: np.ndarray) -> list[tuple[np.ndarray, str, float]]:
        """
        Detect and recognize all text in an image.

        Parameters
        ----------
        image : np.ndarray
            The image in HWC, BGR, uint8 format.

        Returns
        -------
        list[tuple[np.ndarray, str, float]]
            ``(box, text, score)`` per text line in reading order, where ``box``
            is the (4, 2) float32 corner points (TL, TR, BR, BL) in the
            coordinates of ``image``.

        """
        raw_h, raw_w = image.shape[:2]
        img, ratio_h, ratio_w = _fit_side(image, self._max_side_len, shrink=True)
        if min(img.shape[:2]) < self._min_side_len:
            img, ratio_h, ratio_w = _fit_side(img, self._min_side_len, shrink=False)

        h, w = img.shape[:2]
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

        out_boxes = boxes.copy()
        out_boxes[:, :, 1] -= pad
        out_boxes[:, :, 0] *= ratio_w
        out_boxes[:, :, 1] *= ratio_h
        out_boxes = np.maximum(out_boxes, 0)
        out_boxes[:, :, 0] = np.minimum(out_boxes[:, :, 0], raw_w)
        out_boxes[:, :, 1] = np.minimum(out_boxes[:, :, 1], raw_h)
        return [
            (box, text, score)
            for box, (text, score) in zip(out_boxes, lines)
            if score >= self._text_score
        ]

    def __call__(self: Self, image: np.ndarray) -> list[tuple[np.ndarray, str, float]]:
        """
        Detect and recognize all text in an image, see :meth:`run`.

        Parameters
        ----------
        image : np.ndarray
            The image in HWC, BGR, uint8 format.

        Returns
        -------
        list[tuple[np.ndarray, str, float]]
            ``(box, text, score)`` per text line.

        """
        return self.run(image)
