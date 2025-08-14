# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import contextlib
import math
from typing import TYPE_CHECKING

import numpy as np

from trtutils.core._bindings import Binding, create_binding
from trtutils.core._kernels import Kernel
from trtutils.core._memory import (
    memcpy_device_to_host_async,
    memcpy_host_to_device_async,
)
from trtutils.core._stream import create_stream, destroy_stream, stream_synchronize
from trtutils.image.kernels import RESCALE_V10_DETECTIONS, RESCALE_EFF_NMS_DETECTIONS

from ._abc import DetectionPostprocessor

if TYPE_CHECKING:
    from typing_extensions import Self

    with contextlib.suppress(ImportError):
        try:
            import cuda.bindings.runtime as cudart
        except (ImportError, ModuleNotFoundError):
            from cuda import cudart

_EFF_NMS_OUTPUTS = 4


class _RescaleKernel:
    def __init__(
        self: Self,
        topk: int,
        bbox_size: int,
        threads: tuple[int, int, int] | None = None,
        *,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
    ) -> None:
        self._topk = topk
        self._bbox_size = bbox_size

        # handle memory flags
        self._pagelocked_mem = pagelocked_mem if pagelocked_mem is not None else True
        self._unified_mem = unified_mem

        dummy_input: np.ndarray = np.zeros(
            (1, self._topk, self._bbox_size),
            dtype=np.float32,
        )
        self._input_binding: Binding = create_binding(
            dummy_input,
            is_input=True,
            pagelocked_mem=self._pagelocked_mem,
            unified_mem=self._unified_mem,
        )
        self._output_binding: Binding = create_binding(
            dummy_input,
            is_output=True,
            pagelocked_mem=self._pagelocked_mem,
            unified_mem=self._unified_mem,
        )
        dummy_mask: np.ndarray = np.zeros(
            (self._topk,),
            dtype=bool,
        )
        self._mask_binding: Binding = create_binding(
            dummy_mask,
            is_output=True,
            pagelocked_mem=self._pagelocked_mem,
            unified_mem=self._unified_mem,
        )

        # block and thread info
        self._num_threads: tuple[int, int, int] = threads or (256, 1, 1)
        self._num_blocks: tuple[int, int, int] = (
            math.ceil(self._topk / self._num_threads[0]),
            1,
            1,
        )

        # kernel
        self._kernel = Kernel(*RESCALE_V10_DETECTIONS)

    def __del__(self: Self) -> None:
        with contextlib.suppress(AttributeError):
            del self._input_binding
        with contextlib.suppress(AttributeError):
            del self._output_binding
        with contextlib.suppress(AttributeError):
            del self._mask_binding

    def launch(
        self: Self,
        outputs: list[np.ndarray],
        ratios: tuple[float, float],
        padding: tuple[float, float],
        conf_thres: float,
        stream: cudart.cudaStream_t,
        *,
        no_copy: bool | None = None,
    ) -> np.ndarray:
        # extern "C" __global__
        # void rescaleDetections(
        #     const float* __restrict__ boxes,
        #     float* __restrict__ output,
        #     bool* __restrict__ mask,
        #     const int topk,
        #     const int bboxSize,
        #     const float widthScale,
        #     const float heightScale,
        #     const float widthOffset,
        #     const float heightOffset
        # ) {
        is_efficient_nms = len(outputs) == _EFF_NMS_OUTPUTS
        if not is_efficient_nms:
            boxes = outputs[0]
        else:
            num_dets = int(outputs[0][0])
            boxes = outputs[1]

        # copy boxes to device
        memcpy_host_to_device_async(
            self._input_binding.allocation,
            boxes,
            stream=stream,
        )
        args = self._kernel.create_args(
            self._input_binding.allocation,
            self._output_binding.allocation,
            self._mask_binding.allocation,
            # if not efficientNMS, use topk, otherwise use num_dets
            # since efficientNMS is sorted, we can use num_dets as topk
            self._topk if not is_efficient_nms else num_dets,
            self._bbox_size,
            conf_thres,
            ratios[0],
            ratios[1],
            padding[0],
            padding[1],
        )
        self._kernel.call(
            num_blocks=self._blocks, num_threads=self._threads, stream=stream, args=args
        )
        memcpy_device_to_host_async(
            self._output_binding.host_allocation,
            self._output_binding.allocation,
            stream=stream,
        )
        memcpy_device_to_host_async(
            self._mask_binding.host_allocation,
            self._mask_binding.allocation,
            stream=stream,
        )
        stream_synchronize(stream)

        new_boxes = (
            self._output_binding.host_allocation
            if no_copy
            else self._output_binding.host_allocation.copy()
        )
        masked_bboxes = new_boxes[self._mask_binding.host_allocation]

        if not is_efficient_nms:
            return [new_boxes[self._mask_binding.host_allocation]]
        num_dets = outputs[0]
        scores = outputs[2]
        class_ids = outputs[3]
        masked_scores = scores[self._mask_binding.host_allocation]
        masked_class_ids = class_ids[self._mask_binding.host_allocation]
        if no_copy:
            return [num_dets, masked_bboxes, masked_scores, masked_class_ids]
        return [
            num_dets.copy(),
            masked_bboxes,
            masked_scores.copy(),
            masked_class_ids.copy(),
        ]


class CUDADetectionPostprocessor(DetectionPostprocessor):
    """CUDA-based detection postprocessor."""

    def __init__(
        self: Self,
        topk: int,
        bbox_size: int,
        stream: cudart.cudaStream_t | None = None,
        *,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
    ) -> None:
        """
        Create a CUDA-based detection postprocessor.

        Parameters
        ----------
        topk : int
            The number of bounding boxes in the output.
        bbox_size : int
            The number of elements in each bounding box.
            There are 4 elements in EfficientNMS and 6 in YOLOv10.
        stream : cudart.cudaStream_t, optional
            The CUDA stream to use for postprocessing execution.
            If not provided, the postprocessor will use its own stream.
        pagelocked_mem : bool, optional
            Whether or not to allocate output memory as pagelocked.
            By default, pagelocked memory will be used.
        unified_mem : bool, optional
            Whether or not the system has unified memory.
            If True, use cudaHostAllocMapped to take advantage of unified memory.
            By default None, which means the default host allocation will be used.

        """
        self._topk = topk
        self._bbox_size = bbox_size

        # handle memory flags
        self._pagelocked_mem = pagelocked_mem if pagelocked_mem is not None else True
        self._unified_mem = unified_mem

        # handle stream
        self._stream: cudart.cudaStream_t
        self._own_stream = False
        if stream is not None:
            self._stream = stream
        else:
            self._stream = create_stream()
            self._own_stream = True

        # allocate the rescaling kernel
        self._rescale_kernel = _RescaleKernel(
            self._topk,
            self._bbox_size,
            pagelocked_mem=self._pagelocked_mem,
            unified_mem=self._unified_mem,
        )

    def __del__(self: Self) -> None:
        with contextlib.suppress(AttributeError):
            del self._rescale_kernel
        with contextlib.suppress(AttributeError):
            if self._own_stream:
                destroy_stream(self._stream)

    def postprocess(
        self: Self,
        outputs: list[np.ndarray],
        ratios: tuple[float, float],
        padding: tuple[float, float],
        conf_thres: float,
    ) -> list[np.ndarray]: ...
