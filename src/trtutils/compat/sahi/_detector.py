# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: ARG002, B006
# mypy: disable-error-code="import-untyped, import-not-found"
from __future__ import annotations

from typing import TYPE_CHECKING

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction

from trtutils.image import Detector

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import Self


class TRTDetectionModel(DetectionModel):  # type: ignore[misc]
    def check_dependencies(self: Self) -> None:
        pass

    def load_model(self: Self) -> None:
        self.model = Detector(self.model_path, warmup=True)

    @property
    def has_mask(self: Self) -> bool:
        return False

    def perform_inference(
        self: Self, image: np.ndarray, image_size: int | None = None
    ) -> None:
        self._original_predictions = self.model.end2end(image)

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: list[list[int]] | None = [[0, 0]],
        full_shape_list: list[list[int]] | None = None,
    ) -> None:
        if shift_amount_list is None:
            shift_amount_list = [[0, 0]]
        if isinstance(shift_amount_list[0], (list, tuple)):
            shift_amount = shift_amount_list[0]
        else:
            shift_amount = shift_amount_list

        if full_shape_list is None:
            full_shape = None
        else:
            full_shape = (
                full_shape_list[0]
                if isinstance(full_shape_list[0], (list, tuple))
                else full_shape_list
            )

        predictions: list[ObjectPrediction] = []

        num_dets = len(self._original_predictions)

        if num_dets == 0:
            self._object_prediction_list_per_image = [predictions]
            return

        for (x1, y1, x2, y2), score, class_id in self._original_predictions:
            if score < self.confidence_threshold:
                continue

            x1_i, y1_i, x2_i, y2_i = map(int, (x1, y1, x2, y2))

            if full_shape is not None:
                h, w = full_shape
                x1_i, x2_i = max(0, x1_i), min(w, x2_i)
                y1_i, y2_i = max(0, y1_i), min(h, y2_i)
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
