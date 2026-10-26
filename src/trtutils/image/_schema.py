# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from typing_extensions import Self

    from trtutils._engine import TRTEngine

_SchemaT = TypeVar("_SchemaT", bound=Enum)


class InputSchema(Enum):
    # YOLO-X, v7, v8, v9, v10, v11, v12, v13
    YOLO = ("images",)
    # RF-DETR
    RF_DETR = ("input",)
    # DEIM v1/v2, RT-DETR v1/v2, D-FINE
    RT_DETR = ("images", "orig_target_sizes")
    # RT-DETR v3 (PaddlePaddle export format)
    RT_DETR_V3 = ("im_shape", "image", "scale_factor")

    @classmethod
    def names(cls: type[Self]) -> list[str]:
        """
        Get the names of the input schema enums.

        Returns
        -------
        list[str]
            The names of the input schemas.

        """
        return list(cls.__members__.keys())

    @property
    def uses_image_size(self: Self) -> bool:
        """Whether this schema requires an original image size input."""
        return self in (InputSchema.RT_DETR, InputSchema.RT_DETR_V3)

    @property
    def uses_scale_factor(self: Self) -> bool:
        """Whether this schema requires a scale factor input."""
        return self == InputSchema.RT_DETR_V3


class HandInteractionOutputSchema(Enum):
    # HOI-DETR
    HOI_DETR = ("boxes", "scores", "labels", "pair_probs")
    # Hands23, adds the hand side head
    HANDS23 = ("boxes", "scores", "labels", "pair_probs", "side")


# "*" matches any other outputs, named members win over it
class ClassifierOutputSchema(Enum):
    # torchvision classifiers, logits (B, C)
    TORCHVISION = ("output",)
    # any other engine, e.g. custom exports
    GENERIC = ("*",)


class DepthEstimatorOutputSchema(Enum):
    # Depth-Anything v1/v2 (B, H, W) and v3 (B, 1, H, W)
    DEPTH_ANYTHING = ("output",)
    # any other engine, e.g. custom exports
    GENERIC = ("*",)


class TextDetectorOutputSchema(Enum):
    # PP-OCRv4 DB detection, probability map (B, 1, H, W)
    PADDLE = ("sigmoid_0.tmp_0",)
    # any other engine, e.g. custom exports
    GENERIC = ("*",)


class TextRecognizerOutputSchema(Enum):
    # PP-OCRv4 CTC recognition, per-timestep class probabilities (B, T, C)
    PADDLE = ("softmax_11.tmp_0",)
    # any other engine, e.g. custom exports
    GENERIC = ("*",)


class OutputSchema(Enum):
    # YOLO-X, v7, v8, v9, v11, v12, v13
    EFFICIENT_NMS = ("num_dets", "det_boxes", "det_scores", "det_classes")
    EFFICIENT_NMS_2 = ("num", "boxes", "scores", "classes")
    # YOLO-v10
    YOLO_V10 = ("output0",)
    # RF-DETR
    RF_DETR = ("dets", "labels")
    # RT-DETR v1/v2
    DETR = ("scores", "labels", "boxes")
    # DEIM v1/v2, D-FINE
    DETR_LBS = ("labels", "boxes", "scores")
    # RT-DETR v3 (PaddlePaddle export format)
    RT_DETR_V3 = ("save_infer_model/scale_0.tmp_0", "save_infer_model/scale_1.tmp_0")

    @classmethod
    def names(cls: type[Self]) -> list[str]:
        """
        Get the names of the input/output schema enums.

        Returns
        -------
        list[str]
            The names of the input/output schemas.

        """
        return list(cls.__members__.keys())


class SegmenterOutputSchema(Enum):
    # ultralytics segmentation heads: detections + mask prototypes
    YOLO = ("output0", "output1")


class PoseEstimatorOutputSchema(Enum):
    # ultralytics pose heads
    YOLO = ("output0",)


class OBBDetectorOutputSchema(Enum):
    # ultralytics oriented bounding box heads
    YOLO = ("output0",)


def resolve_output_schema(
    engine: TRTEngine,
    schema_cls: type[_SchemaT],
    override: _SchemaT | str | None = None,
    *,
    inputs: bool = False,
) -> _SchemaT:
    """
    Resolve the output (or input) schema of an engine against the members of a schema enum.

    Parameters
    ----------
    engine : TRTEngine
        The loaded TensorRT engine, its output names are matched ignoring order
        since TensorRT 8.6 does not keep the ONNX output order.
    schema_cls : type[Enum]
        The schema enum, each member's value is a tuple of output names.
        A ``"*"`` entry matches any other outputs, so ``("*",)`` matches every
        engine; members without one are tried first.
    override : Enum, str, optional
        A member or member name to use instead of auto-detection.
        The engine names must still match it.
    inputs : bool, optional
        Match the engine input names instead of the output names.

    Returns
    -------
    Enum
        The matching member of ``schema_cls``.

    Raises
    ------
    ValueError
        If the override is not a member name, or the engine names match no
        candidate member.

    """
    kind = "input" if inputs else "output"
    if isinstance(override, str):
        if override not in schema_cls.__members__:
            err_msg = f"Invalid {kind}_schema string: {override}. "
            err_msg += f"Valid options: {list(schema_cls.__members__)}"
            raise ValueError(err_msg)
        override = schema_cls[override]
    candidates = list(schema_cls) if override is None else [override]
    candidates.sort(key=lambda m: "*" in m.value)
    names = engine.input_names if inputs else engine.output_names
    outputs = sorted(names)
    for member in candidates:
        named = set(member.value) - {"*"}
        if sorted(member.value) == outputs or ("*" in member.value and named <= set(outputs)):
            return member
    err_msg = f"Engine {kind}s {list(names)} match no {schema_cls.__name__}. "
    err_msg += f"Valid options: { {m.name: m.value for m in candidates} }"
    raise ValueError(err_msg)
