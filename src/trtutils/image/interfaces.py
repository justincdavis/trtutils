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
OBBDetectorInterface
    Interface for oriented bounding box detectors.
PoseEstimatorInterface
    Interface for pose estimation models.
SegmenterInterface
    Interface for instance segmentation models.
HandInteractionDetectorInterface
    Interface for hand-object interaction detectors.

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
    from trtutils.image.postprocessors._hand_interaction import HandInteraction
    from trtutils.image.postprocessors._obb import OBBDetection
    from trtutils.image.postprocessors._pose import Pose
    from trtutils.image.postprocessors._segmentation import Segmentation


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


class HandInteractionDetectorInterface(ABC):
    """
    Interface for hand-object interaction detectors.

    Implementations wrap engines following the unified hand-object interaction
    output contract: [boxes (B,K,4), scores (B,K), labels (B,K), pair_probs
    (B,K,K,C), side (B,K)]; side is optional. Postprocessed per-image outputs
    pair hands (label 0) with a first object (label 1) and optionally a second
    object (label 2) into ``HandInteraction`` tuples of the form
    ``((hand_bbox, hand_score), (obj_bbox, obj_score) | None, (second_bbox,
    second_score) | None, side | None, contact | None)``, where each bbox is
    an int ``(x1, y1, x2, y2)`` in original image coordinates.
    """

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

    # get_interactions overloads
    @overload
    @abstractmethod
    def get_interactions(
        self: Self,
        outputs: list[np.ndarray],
        pair_thres: float | None = ...,
        second_pair_thres: float | None = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[HandInteraction]: ...

    @overload
    @abstractmethod
    def get_interactions(
        self: Self,
        outputs: list[list[np.ndarray]],
        pair_thres: float | None = ...,
        second_pair_thres: float | None = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[list[HandInteraction]]: ...

    @abstractmethod
    def get_interactions(
        self: Self,
        outputs: list[np.ndarray] | list[list[np.ndarray]],
        pair_thres: float | None = None,
        second_pair_thres: float | None = None,
        *,
        verbose: bool | None = None,
    ) -> list[HandInteraction] | list[list[HandInteraction]]:
        """Get the hand-object interactions for each image."""

    # end2end overloads
    @overload
    @abstractmethod
    def end2end(
        self: Self,
        images: np.ndarray,
        *,
        conf_thres: float | None = ...,
        pair_thres: float | None = ...,
        second_pair_thres: float | None = ...,
        verbose: bool | None = ...,
    ) -> list[HandInteraction]: ...

    @overload
    @abstractmethod
    def end2end(
        self: Self,
        images: list[np.ndarray],
        *,
        conf_thres: float | None = ...,
        pair_thres: float | None = ...,
        second_pair_thres: float | None = ...,
        verbose: bool | None = ...,
    ) -> list[list[HandInteraction]]: ...

    @abstractmethod
    def end2end(
        self: Self,
        images: np.ndarray | list[np.ndarray],
        *,
        conf_thres: float | None = None,
        pair_thres: float | None = None,
        second_pair_thres: float | None = None,
        verbose: bool | None = None,
    ) -> list[HandInteraction] | list[list[HandInteraction]]:
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


class SegmenterInterface(DetectorInterface):
    """
    Interface for instance segmentation models.

    Extends the detector contract: postprocessed per-image outputs are
    ``[bboxes (N,4), scores (N,), class_ids (N,), masks (N,H,W)]``, where masks
    are ``uint8`` at original image resolution. The extra array is appended, so
    :meth:`get_detections` continues to work unchanged.
    """

    # get_segmentations overloads
    @overload
    @abstractmethod
    def get_segmentations(
        self: Self,
        outputs: list[np.ndarray],
        conf_thres: float | None = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[Segmentation]: ...

    @overload
    @abstractmethod
    def get_segmentations(
        self: Self,
        outputs: list[list[np.ndarray]],
        conf_thres: float | None = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[list[Segmentation]]: ...

    @abstractmethod
    def get_segmentations(
        self: Self,
        outputs: list[np.ndarray] | list[list[np.ndarray]],
        conf_thres: float | None = None,
        *,
        verbose: bool | None = None,
    ) -> list[Segmentation] | list[list[Segmentation]]:
        """Get the segmentations for each image."""


class PoseEstimatorInterface(DetectorInterface):
    """
    Interface for pose estimation models.

    Extends the detector contract: postprocessed per-image outputs are
    ``[bboxes (N,4), scores (N,), class_ids (N,), keypoints (N,K,3)]``, where
    each keypoint is ``(x, y, visibility)`` in original image coordinates. The
    extra array is appended, so :meth:`get_detections` continues to work unchanged.
    """

    # get_poses overloads
    @overload
    @abstractmethod
    def get_poses(
        self: Self,
        outputs: list[np.ndarray],
        conf_thres: float | None = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[Pose]: ...

    @overload
    @abstractmethod
    def get_poses(
        self: Self,
        outputs: list[list[np.ndarray]],
        conf_thres: float | None = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[list[Pose]]: ...

    @abstractmethod
    def get_poses(
        self: Self,
        outputs: list[np.ndarray] | list[list[np.ndarray]],
        conf_thres: float | None = None,
        *,
        verbose: bool | None = None,
    ) -> list[Pose] | list[list[Pose]]:
        """Get the poses for each image."""


class OBBDetectorInterface(DetectorInterface):
    """
    Interface for oriented bounding box detectors.

    Extends the detector contract: postprocessed per-image outputs are
    ``[bboxes (N,4), scores (N,), class_ids (N,), rboxes (N,5)]``. ``rboxes`` is
    the authoritative ``(cx, cy, w, h, angle)`` with angle in radians; ``bboxes``
    is the derived enclosing axis-aligned box, so :meth:`get_detections`
    continues to work unchanged.
    """

    # get_obb_detections overloads
    @overload
    @abstractmethod
    def get_obb_detections(
        self: Self,
        outputs: list[np.ndarray],
        conf_thres: float | None = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[OBBDetection]: ...

    @overload
    @abstractmethod
    def get_obb_detections(
        self: Self,
        outputs: list[list[np.ndarray]],
        conf_thres: float | None = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[list[OBBDetection]]: ...

    @abstractmethod
    def get_obb_detections(
        self: Self,
        outputs: list[np.ndarray] | list[list[np.ndarray]],
        conf_thres: float | None = None,
        *,
        verbose: bool | None = None,
    ) -> list[OBBDetection] | list[list[OBBDetection]]:
        """Get the oriented bounding box detections for each image."""
