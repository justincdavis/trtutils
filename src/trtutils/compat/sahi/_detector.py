# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: ARG002, B006
# mypy: disable-error-code="import-untyped, import-not-found"
from __future__ import annotations

from typing import TYPE_CHECKING

import nvtx
from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction

from trtutils._flags import FLAGS
from trtutils.image import Detector

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import Self


class TRTDetectionModel(DetectionModel):
    def check_dependencies(self: Self) -> None:
        pass

    def load_model(self: Self) -> None:
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range("compat::sahi::trt_detection_model::load_model")
        if self.model_path is None:
            err_msg = "model_path must be set before loading model"
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()
            raise ValueError(err_msg)
        self.model = Detector(self.model_path, warmup=True)
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()

    @property
    def has_mask(self: Self) -> bool:
        return False

    def perform_inference(self: Self, image: np.ndarray, image_size: int | None = None) -> None:
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range("compat::sahi::trt_detection_model::perform_inference")
        if self.model is None:
            err_msg = "Model must be loaded before performing inference"
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()
            raise RuntimeError(err_msg)
        self._original_predictions = self.model.end2end([image])[0]
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: list[list[int]] | None = [[0, 0]],
        full_shape_list: list[list[int]] | None = None,
    ) -> None:
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range("compat::sahi::trt_detection_model::create_predictions")
        if shift_amount_list is None:
            shift_amount_list = [[0, 0]]

        # Type narrowing: extract list[int] from list[list[int]] or use list[int] directly
        if isinstance(shift_amount_list[0], (list, tuple)) and len(shift_amount_list[0]) == 2:  # noqa: PLR2004
            shift_amount: list[int] = list(shift_amount_list[0])
        elif (
            isinstance(shift_amount_list, list)
            and len(shift_amount_list) == 2  # noqa: PLR2004
            and isinstance(shift_amount_list[0], int)
        ):
            shift_amount = shift_amount_list
        else:
            shift_amount = [0, 0]

        if full_shape_list is None:
            full_shape: list[int] | None = None
        else:
            if isinstance(full_shape_list[0], (list, tuple)) and len(full_shape_list[0]) == 2:  # noqa: PLR2004
                full_shape = list(full_shape_list[0])
            elif (
                isinstance(full_shape_list, list)
                and len(full_shape_list) == 2  # noqa: PLR2004
                and isinstance(full_shape_list[0], int)
            ):
                full_shape = full_shape_list
            else:
                full_shape = None

        predictions: list[ObjectPrediction] = []

        if self._original_predictions is None:
            self._object_prediction_list_per_image = [predictions]
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()
            return

        num_dets = len(self._original_predictions)

        if num_dets == 0:
            self._object_prediction_list_per_image = [predictions]
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()
            return

        for (x1, y1, x2, y2), score, class_id in self._original_predictions:
            if score < self.confidence_threshold:
                continue

            x1_i, y1_i, x2_i, y2_i = int(x1), int(y1), int(x2), int(y2)

            if full_shape is not None:
                h: int = full_shape[0]  # type: ignore[assignment]
                w: int = full_shape[1]  # type: ignore[assignment]
                x1_i = max(0, x1_i)
                x2_i = min(w, x2_i)
                y1_i = max(0, y1_i)
                y2_i = min(h, y2_i)
            if x2_i <= x1_i or y2_i <= y1_i:
                continue

            predictions.append(
                ObjectPrediction(
                    bbox=[x1_i, y1_i, x2_i, y2_i],
                    score=float(score),
                    category_id=int(class_id),
                    category_name=str(int(class_id)),
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
            )
        self._object_prediction_list_per_image = [predictions]
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # create_predictions
