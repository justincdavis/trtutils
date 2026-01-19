# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Define utilities for benchmarking."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from ultralytics import YOLO

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import Self


class UltralyticsTRTModel(DetectionModel):
    def check_dependencies(self: Self) -> None:
        pass

    def load_model(self: Self) -> None:
        self.model = YOLO(self.model_path, task="detect", verbose=False)  
        
        if not self.category_mapping:
            names = getattr(getattr(self.model, "model", None), "names", None)
            if isinstance(names, dict):
                self.category_mapping = {str(k): v for k, v in names.items()}
            else:
                num_classes = getattr(getattr(self.model, "model", None), "nc", 1000)
                self.category_mapping = {str(i): str(i) for i in range(num_classes)}

    @property
    def has_mask(self: Self) -> bool:
        return False

    def num_categories(self: Self) -> int:
        return len(self.category_mapping)

    def perform_inference(self: Self, image: np.ndarray, image_size: int | None = None) -> None:
        self._original_predictions = self.model.predict(
            image,
            conf=self.confidence_threshold,
            verbose=False,
        )
    
    def _create_object_prediction_list_from_original_predictions(
        self: Self,
        shift_amount_list: list[list[int]] | None = [[0, 0]],
        full_shape_list: list[list[int]] | None = None,
    ) -> None:
        if isinstance(shift_amount_list[0], (list, tuple)):
            shift_amount = shift_amount_list[0]
        else:
            shift_amount = shift_amount_list

        if full_shape_list is None:
            full_shape = None
        else:
            full_shape = full_shape_list[0] if isinstance(full_shape_list[0], (list, tuple)) else full_shape_list

        predictions: list[ObjectPrediction] = []

        result = self._original_predictions[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            self._object_prediction_list_per_image = [predictions]
            return

        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy()

        for (x1, y1, x2, y2), score, class_id in zip(xyxy, scores, class_ids):
            if score < self.confidence_threshold:
                continue

            x1_i, y1_i, x2_i, y2_i = map(int, (x1, y1, x2, y2))

            if full_shape is not None:
                h, w = full_shape
                x1_i, x2_i = max(0, x1_i), min(w, x2_i)
                y1_i, y2_i = max(0, y1_i), min(h, y2_i)
            if x2_i <= x1_i or y2_i <= y1_i:
                continue
            category_name = self.category_mapping.get(str(int(class_id)), str(int(class_id)))

            predictions.append(
                ObjectPrediction(
                    bbox=[x1_i, y1_i, x2_i, y2_i],
                    score=float(score),
                    category_id=int(class_id),
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
            )
        self._object_prediction_list_per_image = [predictions]
