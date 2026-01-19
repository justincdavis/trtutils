# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from trtutils._log import LOG

if TYPE_CHECKING:
    from typing import ClassVar
    from typing_extensions import Self

    from trtutils._engine import TRTEngine


class InputSchema(Enum):
    # YOLO-X, v7, v8, v9, v10, v11, v12, v13
    YOLO: ClassVar[list[str]] = ["images"]
    # RF-DETR
    RF_DETR: ClassVar[list[str]] = ["input"]
    # DEIM v1/v2, RT-DETR v1/v2, D-FINE
    RT_DETR: ClassVar[list[str]] = ["images", "orig_target_sizes"]
    # RT-DETR v3
    RT_DETR_V3: ClassVar[list[str]] = ["image", "im_shape", "scale_factor"]

    @property
    def names(self: Self) -> list[str]:
        """
        Get the names of the input schema enums.

        Returns
        -------
        list[str]
            The names of the input schemas.
        """
        return list(self.__members__.keys())


class OutputSchema(Enum):
    # YOLO-X, v7, v8, v9, v11, v12, v13
    EFFICIENT_NMS: ClassVar[list[str]] = ["num_dets", "det_boxes", "det_scores", "det_classes"]
    EFFICIENT_NMS_2: ClassVar[list[str]] = ["num", "boxes", "scores", "classes"]
    # YOLO-v10
    YOLO_V10: ClassVar[list[str]] = ["output0"]
    # RF-DETR
    RF_DETR: ClassVar[list[str]] = ["dets", "labels"]
    # DEIM v1/v2, RT-DETR v1/v2/v3, D-FINE
    DETR: ClassVar[list[str]] = ["scores", "labels", "boxes"]

    @property
    def names(self: Self) -> list[str]:
        """
        Get the names of the input/output schema enums.

        Returns
        -------
        list[str]
            The names of the input/output schemas.
        """
        return list(self.__members__.keys())


def get_detector_io_schema(
    engine: TRTEngine,
) -> tuple[InputSchema, OutputSchema]:
    # solve for the input scheme
    input_schema: InputSchema
    if engine.input_names == InputSchema.YOLO.value:
        input_schema = InputSchema.YOLO
    elif engine.input_names == InputSchema.RF_DETR.value:
        input_schema = InputSchema.RF_DETR
    elif engine.input_names == InputSchema.RT_DETR.value:
        input_schema = InputSchema.RT_DETR
    elif engine.input_names == InputSchema.RT_DETR_V3.value:
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
        engine.output_names == OutputSchema.EFFICIENT_NMS.value
        or engine.output_names == OutputSchema.EFFICIENT_NMS_2.value
    ):
        output_schema = OutputSchema.EFFICIENT_NMS
    elif engine.output_names == OutputSchema.YOLO_V10.value:
        output_schema = OutputSchema.YOLO_V10
    elif engine.output_names == OutputSchema.RF_DETR.value:
        output_schema = OutputSchema.RF_DETR
    elif engine.output_names == OutputSchema.DETR.value:
        output_schema = OutputSchema.DETR
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
            output_schema = OutputSchema.RF_DETR
        elif len(engine.output_spec) == len(OutputSchema.DETR.value):
            output_schema = OutputSchema.DETR
        else:
            err_msg = f"Expected 1, 2, 3, or 4 outputs, found {len(engine.output_spec)}"
            raise ValueError(err_msg)

    return (input_schema, output_schema)
