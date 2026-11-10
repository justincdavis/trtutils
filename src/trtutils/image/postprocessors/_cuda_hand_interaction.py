# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""CUDA postprocessing for hand-object interaction outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from trtutils.core._buffer import Buffer
from trtutils.image.kernels import HOI_POST, index_kind

from ._cuda import CUDAPostprocessor, split_detections

if TYPE_CHECKING:
    from typing_extensions import Self

    from trtutils.compat._libs import cudart

_WITH_SIDE = 5


class CUDAHoiPostprocessor(CUDAPostprocessor):
    """
    Unletterbox and class-aware NMS for a hand-object interaction batch.

    Same kept rows as ``postprocess_hand_interactions``.
    """

    _FILE = HOI_POST

    def __init__(self: Self, stream: cudart.cudaStream_t) -> None:
        super().__init__(stream)
        self._post = self._kernel("hoiPost")

    def _allocate(self: Self, batch: int, k: int, classes: int) -> None:
        self._rows = Buffer.device((batch, k, 6), np.float32)
        self._src = Buffer.device((batch, k), np.int32)
        self._order = Buffer.device((batch, k), np.int32)
        self._keep = Buffer.device((batch, k), np.int32)
        self._dets = Buffer.host_mapped((batch, k, 6), np.float32)
        self._pairs = Buffer.host_mapped((batch, k, k, classes), np.float32)
        self._side = Buffer.host_mapped((batch, k, 1), np.int64)
        self._counts = Buffer.host_mapped((batch,), np.int32)

    def __call__(
        self: Self,
        outputs: list[Buffer],
        ratios: list[tuple[float, float]],
        padding: list[tuple[float, float]],
        conf_thres: float = 0.3,
        nms_iou_thres: float = 0.5,
        *,
        input_size: tuple[int, int] | None = None,
    ) -> list[list[np.ndarray]]:
        """
        Match ``postprocess_hand_interactions`` for device outputs.

        Parameters
        ----------
        outputs : list[Buffer]
            Device ``[boxes, scores, labels, pair_probs, (side)]``.
        ratios : list[tuple[float, float]]
            Preprocessing resize ratios per image.
        padding : list[tuple[float, float]]
            Preprocessing padding per image.
        conf_thres : float
            Confidence threshold used both to filter candidates and as the NMS score threshold.
        nms_iou_thres : float
            IoU threshold for class-aware NMS.
        input_size : tuple[int, int] | None
            Model input ``(width, height)``. None clips boxes only at 0.

        Returns
        -------
        list[list[np.ndarray]]
            Per image: ``[bboxes, scores, labels, pair_probs, side]``.

        """
        boxes, scores, labels, pairs = outputs[:4]
        side = outputs[4] if len(outputs) == _WITH_SIDE else None
        batch, k, _ = boxes.shape
        classes = int(pairs.shape[-1])
        if self._resized(batch, k, classes):
            self._allocate(batch, k, classes)
        self._launch(
            self._post,
            (batch, 1, 1),
            (1, 1, 1),
            boxes.device_ptr,
            scores.device_ptr,
            labels.device_ptr,
            index_kind(labels.dtype),
            pairs.device_ptr,
            0 if side is None else side.device_ptr,
            0 if side is None else index_kind(side.dtype),
            k,
            classes,
            self._letterbox(ratios, padding, input_size),
            float(conf_thres),
            float(nms_iou_thres),
            self._rows.device_ptr,
            self._src.device_ptr,
            self._order.device_ptr,
            self._keep.device_ptr,
            self._dets.device_ptr,
            self._pairs.device_ptr,
            self._side.device_ptr,
            self._counts.device_ptr,
        )
        self._sync()
        out_side = self._side.array if side is not None else np.zeros((batch, k, 0), int)
        results = split_detections(self._dets.array, self._counts.array, out_side)
        for i, image in enumerate(results):
            n = len(image[1])
            image.insert(3, self._pairs.array[i, :n, :n].copy())
        return results
