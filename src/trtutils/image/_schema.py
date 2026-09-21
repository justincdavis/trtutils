# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from trtutils._log import LOG

if TYPE_CHECKING:
    from typing_extensions import Self

    from trtutils._engine import TRTEngine


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

    @property
    def orig_size_dtype(self: Self) -> np.dtype:
        """The dtype for the original size input tensor."""
        return np.dtype(np.float32) if self == InputSchema.RT_DETR_V3 else np.dtype(np.int32)


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


def get_detector_io_schema(
    engine: TRTEngine,
) -> tuple[InputSchema, OutputSchema]:
    input_names = tuple(engine.input_names)
    output_names = tuple(engine.output_names)
    # solve for the input scheme
    input_schema: InputSchema
    if input_names == InputSchema.YOLO.value:
        input_schema = InputSchema.YOLO
    elif input_names == InputSchema.RF_DETR.value:
        input_schema = InputSchema.RF_DETR
    elif input_names == InputSchema.RT_DETR.value:
        input_schema = InputSchema.RT_DETR
    elif input_names == InputSchema.RT_DETR_V3.value:
        input_schema = InputSchema.RT_DETR_V3
    else:
        warn_msg = "Could not determine input schema directly from input names. "
        warn_msg += f"Input names: {engine.input_names}, "
        warn_msg += f"Input scheme: {engine.input_spec}. "
        warn_msg += "Attemping input schema solve from input spec length."
        LOG.warning(warn_msg)
        if len(engine.input_spec) == len(InputSchema.YOLO.value):
            input_schema = InputSchema.YOLO
        elif len(engine.input_spec) == len(InputSchema.RT_DETR.value):
            input_schema = InputSchema.RT_DETR
        elif len(engine.input_spec) == len(InputSchema.RT_DETR_V3.value):
            input_schema = InputSchema.RT_DETR_V3
        else:
            err_msg = f"Expected 1, 2, or 3 inputs, found {len(engine.input_spec)}"
            raise ValueError(err_msg)

    # solve for the output schema
    output_schema: OutputSchema
    if (
        output_names == OutputSchema.EFFICIENT_NMS.value
        or output_names == OutputSchema.EFFICIENT_NMS_2.value
    ):
        output_schema = OutputSchema.EFFICIENT_NMS
    elif output_names == OutputSchema.YOLO_V10.value:
        output_schema = OutputSchema.YOLO_V10
    elif output_names == OutputSchema.RF_DETR.value:
        output_schema = OutputSchema.RF_DETR
    elif output_names == OutputSchema.DETR.value:
        output_schema = OutputSchema.DETR
    elif output_names == OutputSchema.DETR_LBS.value:
        output_schema = OutputSchema.DETR_LBS
    elif output_names == OutputSchema.RT_DETR_V3.value:
        output_schema = OutputSchema.RT_DETR_V3
    else:
        err_msg = "Could not determine output schema directly from output names. "
        err_msg += f"Output names: {engine.output_names}, "
        err_msg += f"Output scheme: {engine.output_spec}. "
        err_msg += "Attemping output schema solve from output spec length."
        LOG.warning(err_msg)
        if len(engine.output_spec) == len(OutputSchema.EFFICIENT_NMS.value):
            output_schema = OutputSchema.EFFICIENT_NMS
        elif len(engine.output_spec) == len(OutputSchema.YOLO_V10.value):
            output_schema = OutputSchema.YOLO_V10
        elif len(engine.output_spec) == len(OutputSchema.RF_DETR.value):
            # Distinguish RT_DETR_V3 from RF_DETR by checking for PaddlePaddle naming
            if any("scale" in name for name in output_names):
                output_schema = OutputSchema.RT_DETR_V3
            else:
                output_schema = OutputSchema.RF_DETR
        elif len(engine.output_spec) == len(OutputSchema.DETR.value):
            output_schema = OutputSchema.DETR
        else:
            err_msg = f"Expected 1, 2, 3, or 4 outputs, found {len(engine.output_spec)}"
            raise ValueError(err_msg)

    return (input_schema, output_schema)


def resolve_detector_schemas(
    engine: TRTEngine,
    input_schema: InputSchema | str | None = None,
    output_schema: OutputSchema | str | None = None,
) -> tuple[InputSchema, OutputSchema]:
    """
    Resolve input/output schemas from overrides or auto-detection.

    Parameters
    ----------
    engine : TRTEngine
        The loaded TensorRT engine to auto-detect schemas from.
    input_schema : InputSchema, str, optional
        Override for the input schema. Can be an enum value, a string
        matching the enum name, or None for auto-detection.
    output_schema : OutputSchema, str, optional
        Override for the output schema. Can be an enum value, a string
        matching the enum name, or None for auto-detection.

    Returns
    -------
    tuple[InputSchema, OutputSchema]
        The resolved input and output schemas.

    Raises
    ------
    ValueError
        If an input or output schema string is invalid.

    """
    # auto-detect schemas from engine if not provided
    if input_schema is None or output_schema is None:
        auto_input, auto_output = get_detector_io_schema(engine)
    else:
        auto_input = None
        auto_output = None

    # resolve input schema
    resolved_input: InputSchema
    if input_schema is None:
        if auto_input is None:
            err_msg = "Input schema could not be determined from the engine."
            raise ValueError(err_msg)
        resolved_input = auto_input
    elif isinstance(input_schema, str):
        if input_schema not in InputSchema.names():
            err_msg = f"Invalid input_schema string: {input_schema}. "
            err_msg += f"Valid options: {InputSchema.names()}"
            raise ValueError(err_msg)
        resolved_input = InputSchema[input_schema]
    else:
        resolved_input = input_schema

    # resolve output schema
    resolved_output: OutputSchema
    if output_schema is None:
        if auto_output is None:
            err_msg = "Output schema could not be determined from the engine."
            raise ValueError(err_msg)
        resolved_output = auto_output
    elif isinstance(output_schema, str):
        if output_schema not in OutputSchema.names():
            err_msg = f"Invalid output_schema string: {output_schema}. "
            err_msg += f"Valid options: {OutputSchema.names()}"
            raise ValueError(err_msg)
        resolved_output = OutputSchema[output_schema]
    else:
        resolved_output = output_schema

    return (resolved_input, resolved_output)


class SegmentationOutputSchema(Enum):
    # dense prediction head plus mask prototypes
    YOLO = ("output0", "output1")

    @classmethod
    def names(cls: type[Self]) -> list[str]:
        """
        Get the names of the segmentation output schema enums.

        Returns
        -------
        list[str]
            The names of the output schemas.

        """
        return list(cls.__members__.keys())


class PoseOutputSchema(Enum):
    # dense prediction head with per-detection keypoints
    YOLO = ("output0",)

    @classmethod
    def names(cls: type[Self]) -> list[str]:
        """
        Get the names of the pose output schema enums.

        Returns
        -------
        list[str]
            The names of the output schemas.

        """
        return list(cls.__members__.keys())


class OBBOutputSchema(Enum):
    # dense prediction head with a trailing angle channel
    YOLO = ("output0",)

    @classmethod
    def names(cls: type[Self]) -> list[str]:
        """
        Get the names of the OBB output schema enums.

        Returns
        -------
        list[str]
            The names of the output schemas.

        """
        return list(cls.__members__.keys())


_TaskOutputSchema = TypeVar(
    "_TaskOutputSchema",
    SegmentationOutputSchema,
    PoseOutputSchema,
    OBBOutputSchema,
)


def _resolve_input_schema(
    engine: TRTEngine,
    override: InputSchema | str | None,
) -> InputSchema:
    if isinstance(override, InputSchema):
        return override
    if isinstance(override, str):
        if override not in InputSchema.names():
            err_msg = f"Invalid input_schema string: {override}. "
            err_msg += f"Valid options: {InputSchema.names()}"
            raise ValueError(err_msg)
        return InputSchema[override]

    input_names = tuple(engine.input_names)
    for member in InputSchema:
        if input_names == member.value:
            return member
    for member in InputSchema:
        if len(engine.input_spec) == len(member.value):
            return member
    err_msg = f"Could not determine input schema from input names: {engine.input_names}"
    raise ValueError(err_msg)


def _resolve_output_schema(
    engine: TRTEngine,
    schema_cls: type[_TaskOutputSchema],
    override: _TaskOutputSchema | str | None,
) -> _TaskOutputSchema:
    if isinstance(override, schema_cls):
        return override
    if isinstance(override, str):
        valid = list(schema_cls.__members__)
        if override not in valid:
            err_msg = f"Invalid output_schema string: {override}. "
            err_msg += f"Valid options: {valid}"
            raise ValueError(err_msg)
        return schema_cls[override]

    output_names = tuple(engine.output_names)
    for member in schema_cls:
        if output_names == member.value:
            return member
    for member in schema_cls:
        if len(engine.output_spec) == len(member.value):
            return member
    err_msg = f"Could not determine output schema from output names: {engine.output_names}"
    raise ValueError(err_msg)


def resolve_segmentation_schemas(
    engine: TRTEngine,
    input_schema: InputSchema | str | None = None,
    output_schema: SegmentationOutputSchema | str | None = None,
) -> tuple[InputSchema, SegmentationOutputSchema]:
    """
    Resolve segmentation input/output schemas from overrides or auto-detection.

    Parameters
    ----------
    engine : TRTEngine
        The loaded TensorRT engine to auto-detect schemas from.
    input_schema : InputSchema, str, optional
        Override for the input schema, as an enum value or matching enum name.
    output_schema : SegmentationOutputSchema, str, optional
        Override for the output schema, as an enum value or matching enum name.

    Returns
    -------
    tuple[InputSchema, SegmentationOutputSchema]
        The resolved input and output schemas.

    """
    return (
        _resolve_input_schema(engine, input_schema),
        _resolve_output_schema(engine, SegmentationOutputSchema, output_schema),
    )


def resolve_pose_schemas(
    engine: TRTEngine,
    input_schema: InputSchema | str | None = None,
    output_schema: PoseOutputSchema | str | None = None,
) -> tuple[InputSchema, PoseOutputSchema]:
    """
    Resolve pose input/output schemas from overrides or auto-detection.

    Parameters
    ----------
    engine : TRTEngine
        The loaded TensorRT engine to auto-detect schemas from.
    input_schema : InputSchema, str, optional
        Override for the input schema, as an enum value or matching enum name.
    output_schema : PoseOutputSchema, str, optional
        Override for the output schema, as an enum value or matching enum name.

    Returns
    -------
    tuple[InputSchema, PoseOutputSchema]
        The resolved input and output schemas.

    """
    return (
        _resolve_input_schema(engine, input_schema),
        _resolve_output_schema(engine, PoseOutputSchema, output_schema),
    )


def resolve_obb_schemas(
    engine: TRTEngine,
    input_schema: InputSchema | str | None = None,
    output_schema: OBBOutputSchema | str | None = None,
) -> tuple[InputSchema, OBBOutputSchema]:
    """
    Resolve OBB input/output schemas from overrides or auto-detection.

    Parameters
    ----------
    engine : TRTEngine
        The loaded TensorRT engine to auto-detect schemas from.
    input_schema : InputSchema, str, optional
        Override for the input schema, as an enum value or matching enum name.
    output_schema : OBBOutputSchema, str, optional
        Override for the output schema, as an enum value or matching enum name.

    Returns
    -------
    tuple[InputSchema, OBBOutputSchema]
        The resolved input and output schemas.

    """
    return (
        _resolve_input_schema(engine, input_schema),
        _resolve_output_schema(engine, OBBOutputSchema, output_schema),
    )
