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

import numpy as np

if TYPE_CHECKING:
    from typing_extensions import Self


class ClassifierInterface(ABC):
    """Interface for image classifiers."""

    @abstractmethod
    def preprocess(
        self: Self,
        image: np.ndarray,
        resize: str | None = None,
        method: str | None = None,
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
        """
        Preprocess the input.
        """
        pass

    @abstractmethod
    def postprocess(
        self: Self,
        outputs: list[np.ndarray],
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[np.ndarray]:
        """
        Postprocess the outputs.
        """
        pass

    @abstractmethod
    def __call__(
        self: Self,
        image: np.ndarray,
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[np.ndarray]:
        """
        Run the model on input.
        """
        pass

    @abstractmethod
    def run(
        self: Self,
        image: np.ndarray,
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[np.ndarray]:
        """
        Run the model on input.
        """
        pass

    @abstractmethod
    def get_classifications(
        self: Self,
        outputs: list[np.ndarray],
        top_k: int = 5,
        *,
        verbose: bool | None = None,
    ) -> list[tuple[int, float]]:
        """
        Get the classifications of the last output or provided output.
        """
        pass

    @abstractmethod
    def end2end(
        self: Self,
        image: np.ndarray,
        top_k: int = 5,
        *,
        verbose: bool | None = None,
    ) -> list[tuple[int, float]]:
        """
        Perform end to end inference for a model.
        """
        pass


class DetectorInterface(ABC):
    """Interface for image detectors."""

    @abstractmethod
    def preprocess(
        self: Self,
        image: np.ndarray,
        resize: str | None = None,
        method: str | None = None,
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
        """
        Preprocess the input.
        """
        pass

    @abstractmethod
    def postprocess(
        self: Self,
        outputs: list[np.ndarray],
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[np.ndarray]:
        """
        Postprocess the outputs.
        """
        pass

    @abstractmethod
    def run(
        self: Self,
        image: np.ndarray,
        ratios: tuple[float, float] | None = None,
        padding: tuple[float, float] | None = None,
        conf_thres: float | None = None,
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[np.ndarray]:
        """
        Run the model on input.
        """
        pass

    @abstractmethod
    def __call__(
        self: Self,
        image: np.ndarray,
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[np.ndarray]:
        """
        Run the model on input.
        """
        pass

    @abstractmethod
    def get_detections(
        self: Self,
        outputs: list[np.ndarray],
        conf_thres: float | None = None,
        nms_iou_thres: float | None = None,
        *,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
        verbose: bool | None = None,
    ) -> list[tuple[tuple[int, int, int, int], float, int]]:
        """
        Get the detections of the last output or provided output.
        """
        pass

    @abstractmethod
    def end2end(
        self: Self,
        image: np.ndarray,
        conf_thres: float | None = None,
        nms_iou_thres: float | None = None,
        *,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
        verbose: bool | None = None,
    ) -> list[tuple[tuple[int, int, int, int], float, int]]:
        """
        Perform end to end inference for a model.
        """
        pass
