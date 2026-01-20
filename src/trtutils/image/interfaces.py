# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Interaces for the image models.

Classes
-------
ClassifierInterface
    Interface for image classifiers.
DetectorInterface
    Interface for image detectors.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import Self

    from trtutils._engine import TRTEngine
    from trtutils.image._schema import InputSchema, OutputSchema


class ClassifierInterface(ABC):
    """Interface for image classifiers."""

    @property
    @abstractmethod
    def engine(self: Self) -> TRTEngine:
        """Get the underlying TRTEngine."""

    @property
    @abstractmethod
    def name(self: Self) -> str:
        """Get the name of the engine."""

    @property
    @abstractmethod
    def input_shape(self: Self) -> tuple[int, int]:
        """Get the input shape of the model."""

    @property
    @abstractmethod
    def dtype(self: Self) -> np.dtype:
        """Get the dtype required by the model."""

    @abstractmethod
    def preprocess(
        self: Self,
        images: list[np.ndarray],
        resize: str | None = None,
        method: str | None = None,
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]:
        """Preprocess the input images."""

    @abstractmethod
    def postprocess(
        self: Self,
        outputs: list[np.ndarray],
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[np.ndarray] | list[list[np.ndarray]]:
        """Postprocess the outputs."""

    @abstractmethod
    def __call__(
        self: Self,
        images: list[np.ndarray],
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[np.ndarray] | list[list[np.ndarray]]:
        """Run the model on input."""

    @abstractmethod
    def run(
        self: Self,
        images: list[np.ndarray],
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[np.ndarray] | list[list[np.ndarray]]:
        """Run the model on input."""

    @abstractmethod
    def get_classifications(
        self: Self,
        outputs: list[list[np.ndarray]],
        top_k: int = 5,
        *,
        verbose: bool | None = None,
    ) -> list[list[tuple[int, float]]]:
        """Get the classifications for each image."""

    @abstractmethod
    def end2end(
        self: Self,
        images: list[np.ndarray],
        top_k: int = 5,
        *,
        verbose: bool | None = None,
    ) -> list[list[tuple[int, float]]]:
        """Perform end to end inference for a batch of images."""


class DetectorInterface(ABC):
    """Interface for image detectors."""

    @property
    @abstractmethod
    def engine(self: Self) -> TRTEngine:
        """Get the underlying TRTEngine."""

    @property
    @abstractmethod
    def name(self: Self) -> str:
        """Get the name of the engine."""

    @property
    @abstractmethod
    def input_shape(self: Self) -> tuple[int, int]:
        """Get the input shape of the model."""

    @property
    @abstractmethod
    def dtype(self: Self) -> np.dtype:
        """Get the dtype required by the model."""

    @property
    @abstractmethod
    def input_schema(self: Self) -> InputSchema:
        """Get the input schema used by this detector."""

    @property
    @abstractmethod
    def output_schema(self: Self) -> OutputSchema:
        """Get the output schema used by this detector."""

    @abstractmethod
    def preprocess(
        self: Self,
        images: list[np.ndarray],
        resize: str | None = None,
        method: str | None = None,
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]:
        """Preprocess the input images."""

    @abstractmethod
    def postprocess(
        self: Self,
        outputs: list[np.ndarray],
        ratios: list[tuple[float, float]],
        padding: list[tuple[float, float]],
        conf_thres: float | None = None,
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[np.ndarray] | list[list[np.ndarray]]:
        """Postprocess the outputs."""

    @abstractmethod
    def run(
        self: Self,
        images: list[np.ndarray],
        ratios: list[tuple[float, float]] | None = None,
        padding: list[tuple[float, float]] | None = None,
        conf_thres: float | None = None,
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[np.ndarray] | list[list[np.ndarray]]:
        """Run the model on input."""

    @abstractmethod
    def __call__(
        self: Self,
        images: list[np.ndarray],
        ratios: list[tuple[float, float]] | None = None,
        padding: list[tuple[float, float]] | None = None,
        conf_thres: float | None = None,
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[np.ndarray] | list[list[np.ndarray]]:
        """Run the model on input."""

    @abstractmethod
    def get_detections(
        self: Self,
        outputs: list[list[np.ndarray]],
        conf_thres: float | None = None,
        nms_iou_thres: float | None = None,
        *,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
        verbose: bool | None = None,
    ) -> list[list[tuple[tuple[int, int, int, int], float, int]]]:
        """Get the detections for each image."""

    @abstractmethod
    def end2end(
        self: Self,
        images: list[np.ndarray],
        conf_thres: float | None = None,
        nms_iou_thres: float | None = None,
        *,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
        verbose: bool | None = None,
    ) -> list[list[tuple[tuple[int, int, int, int], float, int]]]:
        """Perform end to end inference for a batch of images."""
