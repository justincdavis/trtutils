# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Interaces for the image models.

Classes
-------
ClassifierInterface
    Interface for image classifiers.
DepthEstimatorInterface
    Interface for depth estimators.
DetectorInterface
    Interface for image detectors.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, overload

from typing_extensions import Literal

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

    # preprocess overloads
    @overload
    @abstractmethod
    def preprocess(
        self: Self,
        images: np.ndarray,
        resize: str | None = ...,
        method: str | None = ...,
        *,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]: ...

    @overload
    @abstractmethod
    def preprocess(
        self: Self,
        images: list[np.ndarray],
        resize: str | None = ...,
        method: str | None = ...,
        *,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]: ...

    @abstractmethod
    def preprocess(
        self: Self,
        images: np.ndarray | list[np.ndarray],
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

    # __call__ overloads
    @overload
    @abstractmethod
    def __call__(
        self: Self,
        images: np.ndarray,
        *,
        preprocessed: bool | None = ...,
        postprocess: bool | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    @overload
    @abstractmethod
    def __call__(
        self: Self,
        images: list[np.ndarray],
        *,
        preprocessed: bool | None = ...,
        postprocess: bool | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray] | list[list[np.ndarray]]: ...

    @abstractmethod
    def __call__(
        self: Self,
        images: np.ndarray | list[np.ndarray],
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[np.ndarray] | list[list[np.ndarray]]:
        """Run the model on input."""

    # run overloads - batch input (3 overloads)
    @overload
    @abstractmethod
    def run(
        self: Self,
        images: list[np.ndarray],
        *,
        preprocessed: bool | None = ...,
        postprocess: Literal[False],
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    @overload
    @abstractmethod
    def run(
        self: Self,
        images: list[np.ndarray],
        *,
        preprocessed: bool | None = ...,
        postprocess: Literal[True] | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[list[np.ndarray]]: ...

    @overload
    @abstractmethod
    def run(
        self: Self,
        images: list[np.ndarray],
        *,
        preprocessed: bool | None = ...,
        postprocess: bool | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray] | list[list[np.ndarray]]: ...

    # run overloads - single image input (3 overloads)
    @overload
    @abstractmethod
    def run(
        self: Self,
        images: np.ndarray,
        *,
        preprocessed: bool | None = ...,
        postprocess: Literal[False],
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    @overload
    @abstractmethod
    def run(
        self: Self,
        images: np.ndarray,
        *,
        preprocessed: bool | None = ...,
        postprocess: Literal[True] | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    @overload
    @abstractmethod
    def run(
        self: Self,
        images: np.ndarray,
        *,
        preprocessed: bool | None = ...,
        postprocess: bool | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    @abstractmethod
    def run(
        self: Self,
        images: np.ndarray | list[np.ndarray],
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[np.ndarray] | list[list[np.ndarray]]:
        """Run the model on input."""

    # get_classifications overloads
    @overload
    @abstractmethod
    def get_classifications(
        self: Self,
        outputs: list[np.ndarray],
        top_k: int = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[tuple[int, float]]: ...

    @overload
    @abstractmethod
    def get_classifications(
        self: Self,
        outputs: list[list[np.ndarray]],
        top_k: int = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[list[tuple[int, float]]]: ...

    @abstractmethod
    def get_classifications(
        self: Self,
        outputs: list[np.ndarray] | list[list[np.ndarray]],
        top_k: int = 5,
        *,
        verbose: bool | None = None,
    ) -> list[tuple[int, float]] | list[list[tuple[int, float]]]:
        """Get the classifications for each image."""

    # end2end overloads
    @overload
    @abstractmethod
    def end2end(
        self: Self,
        images: np.ndarray,
        top_k: int = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[tuple[int, float]]: ...

    @overload
    @abstractmethod
    def end2end(
        self: Self,
        images: list[np.ndarray],
        top_k: int = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[list[tuple[int, float]]]: ...

    @abstractmethod
    def end2end(
        self: Self,
        images: np.ndarray | list[np.ndarray],
        top_k: int = 5,
        *,
        verbose: bool | None = None,
    ) -> list[tuple[int, float]] | list[list[tuple[int, float]]]:
        """Perform end to end inference for a batch of images."""


class DepthEstimatorInterface(ABC):
    """Interface for depth estimators."""

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

    # preprocess overloads
    @overload
    @abstractmethod
    def preprocess(
        self: Self,
        images: np.ndarray,
        resize: str | None = ...,
        method: str | None = ...,
        *,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]: ...

    @overload
    @abstractmethod
    def preprocess(
        self: Self,
        images: list[np.ndarray],
        resize: str | None = ...,
        method: str | None = ...,
        *,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]: ...

    @abstractmethod
    def preprocess(
        self: Self,
        images: np.ndarray | list[np.ndarray],
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

    # __call__ overloads
    @overload
    @abstractmethod
    def __call__(
        self: Self,
        images: np.ndarray,
        *,
        preprocessed: bool | None = ...,
        postprocess: bool | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    @overload
    @abstractmethod
    def __call__(
        self: Self,
        images: list[np.ndarray],
        *,
        preprocessed: bool | None = ...,
        postprocess: bool | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray] | list[list[np.ndarray]]: ...

    @abstractmethod
    def __call__(
        self: Self,
        images: np.ndarray | list[np.ndarray],
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[np.ndarray] | list[list[np.ndarray]]:
        """Run the model on input."""

    # run overloads - batch input (3 overloads)
    @overload
    @abstractmethod
    def run(
        self: Self,
        images: list[np.ndarray],
        *,
        preprocessed: bool | None = ...,
        postprocess: Literal[False],
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    @overload
    @abstractmethod
    def run(
        self: Self,
        images: list[np.ndarray],
        *,
        preprocessed: bool | None = ...,
        postprocess: Literal[True] | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[list[np.ndarray]]: ...

    @overload
    @abstractmethod
    def run(
        self: Self,
        images: list[np.ndarray],
        *,
        preprocessed: bool | None = ...,
        postprocess: bool | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray] | list[list[np.ndarray]]: ...

    # run overloads - single image input (3 overloads)
    @overload
    @abstractmethod
    def run(
        self: Self,
        images: np.ndarray,
        *,
        preprocessed: bool | None = ...,
        postprocess: Literal[False],
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    @overload
    @abstractmethod
    def run(
        self: Self,
        images: np.ndarray,
        *,
        preprocessed: bool | None = ...,
        postprocess: Literal[True] | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    @overload
    @abstractmethod
    def run(
        self: Self,
        images: np.ndarray,
        *,
        preprocessed: bool | None = ...,
        postprocess: bool | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    @abstractmethod
    def run(
        self: Self,
        images: np.ndarray | list[np.ndarray],
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[np.ndarray] | list[list[np.ndarray]]:
        """Run the model on input."""

    # get_depth_maps overloads
    @overload
    @abstractmethod
    def get_depth_maps(
        self: Self,
        outputs: list[np.ndarray],
        *,
        verbose: bool | None = ...,
    ) -> np.ndarray: ...

    @overload
    @abstractmethod
    def get_depth_maps(
        self: Self,
        outputs: list[list[np.ndarray]],
        *,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    @abstractmethod
    def get_depth_maps(
        self: Self,
        outputs: list[np.ndarray] | list[list[np.ndarray]],
        *,
        verbose: bool | None = None,
    ) -> np.ndarray | list[np.ndarray]:
        """Get the depth maps for each image."""

    # end2end overloads
    @overload
    @abstractmethod
    def end2end(
        self: Self,
        images: np.ndarray,
        *,
        verbose: bool | None = ...,
    ) -> np.ndarray: ...

    @overload
    @abstractmethod
    def end2end(
        self: Self,
        images: list[np.ndarray],
        *,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    @abstractmethod
    def end2end(
        self: Self,
        images: np.ndarray | list[np.ndarray],
        *,
        verbose: bool | None = None,
    ) -> np.ndarray | list[np.ndarray]:
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

    # preprocess overloads
    @overload
    @abstractmethod
    def preprocess(
        self: Self,
        images: np.ndarray,
        resize: str | None = ...,
        method: str | None = ...,
        *,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]: ...

    @overload
    @abstractmethod
    def preprocess(
        self: Self,
        images: list[np.ndarray],
        resize: str | None = ...,
        method: str | None = ...,
        *,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]: ...

    @abstractmethod
    def preprocess(
        self: Self,
        images: np.ndarray | list[np.ndarray],
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

    # run overloads - batch input (3 overloads)
    @overload
    @abstractmethod
    def run(
        self: Self,
        images: list[np.ndarray],
        ratios: list[tuple[float, float]] | None = ...,
        padding: list[tuple[float, float]] | None = ...,
        conf_thres: float | None = ...,
        *,
        preprocessed: bool | None = ...,
        postprocess: Literal[False],
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    @overload
    @abstractmethod
    def run(
        self: Self,
        images: list[np.ndarray],
        ratios: list[tuple[float, float]] | None = ...,
        padding: list[tuple[float, float]] | None = ...,
        conf_thres: float | None = ...,
        *,
        preprocessed: bool | None = ...,
        postprocess: Literal[True] | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[list[np.ndarray]]: ...

    @overload
    @abstractmethod
    def run(
        self: Self,
        images: list[np.ndarray],
        ratios: list[tuple[float, float]] | None = ...,
        padding: list[tuple[float, float]] | None = ...,
        conf_thres: float | None = ...,
        *,
        preprocessed: bool | None = ...,
        postprocess: bool | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray] | list[list[np.ndarray]]: ...

    # run overloads - single image input (3 overloads)
    @overload
    @abstractmethod
    def run(
        self: Self,
        images: np.ndarray,
        ratios: tuple[float, float] | None = ...,
        padding: tuple[float, float] | None = ...,
        conf_thres: float | None = ...,
        *,
        preprocessed: bool | None = ...,
        postprocess: Literal[False],
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    @overload
    @abstractmethod
    def run(
        self: Self,
        images: np.ndarray,
        ratios: tuple[float, float] | None = ...,
        padding: tuple[float, float] | None = ...,
        conf_thres: float | None = ...,
        *,
        preprocessed: bool | None = ...,
        postprocess: Literal[True] | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    @overload
    @abstractmethod
    def run(
        self: Self,
        images: np.ndarray,
        ratios: tuple[float, float] | None = ...,
        padding: tuple[float, float] | None = ...,
        conf_thres: float | None = ...,
        *,
        preprocessed: bool | None = ...,
        postprocess: bool | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    @abstractmethod
    def run(
        self: Self,
        images: np.ndarray | list[np.ndarray],
        ratios: tuple[float, float] | list[tuple[float, float]] | None = None,
        padding: tuple[float, float] | list[tuple[float, float]] | None = None,
        conf_thres: float | None = None,
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[np.ndarray] | list[list[np.ndarray]]:
        """Run the model on input."""

    # __call__ overloads
    @overload
    @abstractmethod
    def __call__(
        self: Self,
        images: np.ndarray,
        ratios: tuple[float, float] | None = ...,
        padding: tuple[float, float] | None = ...,
        conf_thres: float | None = ...,
        *,
        preprocessed: bool | None = ...,
        postprocess: bool | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    @overload
    @abstractmethod
    def __call__(
        self: Self,
        images: list[np.ndarray],
        ratios: list[tuple[float, float]] | None = ...,
        padding: list[tuple[float, float]] | None = ...,
        conf_thres: float | None = ...,
        *,
        preprocessed: bool | None = ...,
        postprocess: bool | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray] | list[list[np.ndarray]]: ...

    @abstractmethod
    def __call__(
        self: Self,
        images: np.ndarray | list[np.ndarray],
        ratios: tuple[float, float] | list[tuple[float, float]] | None = None,
        padding: tuple[float, float] | list[tuple[float, float]] | None = None,
        conf_thres: float | None = None,
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[np.ndarray] | list[list[np.ndarray]]:
        """Run the model on input."""

    # get_detections overloads
    @overload
    @abstractmethod
    def get_detections(
        self: Self,
        outputs: list[np.ndarray],
        conf_thres: float | None = ...,
        nms_iou_thres: float | None = ...,
        *,
        extra_nms: bool | None = ...,
        agnostic_nms: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[tuple[tuple[int, int, int, int], float, int]]: ...

    @overload
    @abstractmethod
    def get_detections(
        self: Self,
        outputs: list[list[np.ndarray]],
        conf_thres: float | None = ...,
        nms_iou_thres: float | None = ...,
        *,
        extra_nms: bool | None = ...,
        agnostic_nms: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[list[tuple[tuple[int, int, int, int], float, int]]]: ...

    @abstractmethod
    def get_detections(
        self: Self,
        outputs: list[np.ndarray] | list[list[np.ndarray]],
        conf_thres: float | None = None,
        nms_iou_thres: float | None = None,
        *,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
        verbose: bool | None = None,
    ) -> (
        list[tuple[tuple[int, int, int, int], float, int]]
        | list[list[tuple[tuple[int, int, int, int], float, int]]]
    ):
        """Get the detections for each image."""

    # end2end overloads
    @overload
    @abstractmethod
    def end2end(
        self: Self,
        images: np.ndarray,
        conf_thres: float | None = ...,
        nms_iou_thres: float | None = ...,
        *,
        extra_nms: bool | None = ...,
        agnostic_nms: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[tuple[tuple[int, int, int, int], float, int]]: ...

    @overload
    @abstractmethod
    def end2end(
        self: Self,
        images: list[np.ndarray],
        conf_thres: float | None = ...,
        nms_iou_thres: float | None = ...,
        *,
        extra_nms: bool | None = ...,
        agnostic_nms: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[list[tuple[tuple[int, int, int, int], float, int]]]: ...

    @abstractmethod
    def end2end(
        self: Self,
        images: np.ndarray | list[np.ndarray],
        conf_thres: float | None = None,
        nms_iou_thres: float | None = None,
        *,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
        verbose: bool | None = None,
    ) -> (
        list[tuple[tuple[int, int, int, int], float, int]]
        | list[list[tuple[tuple[int, int, int, int], float, int]]]
    ):
        """Perform end to end inference for a batch of images."""
