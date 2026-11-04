# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
CUDA preprocessing for OCR, bit-exact with the OpenCV (4.11, x86) CPU preprocessing.

The kernels mirror ``cv2.resize(INTER_LINEAR)`` and
``cv2.warpPerspective(INTER_CUBIC, BORDER_REPLICATE)`` on 8-bit 3-channel images
fixed point step by step, so the model inputs equal the CPU path's exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import cv2
import numpy as np

from trtutils.compat._libs import cudart
from trtutils.core._buffer import Buffer, MemoryLocation
from trtutils.core._cuda import cuda_call
from trtutils.core._kernels import Kernel, cached_args
from trtutils.core._memory import memcpy_host_to_device_staged_async
from trtutils.image.kernels import OCR_PRE

from ._ocr import (
    text_crop_geometry,
    text_detection_lut,
    text_detection_size,
    text_recognition_lut,
    text_recognition_widths,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self

_ROTATE_ASPECT = 1.5
_TAB_SIZE = 32
_CHANNELS = 3
_THREADS = 256


def is_bgr_frame(image: np.ndarray | Buffer) -> bool:
    """Whether an image is (H, W, 3) uint8, what the kernels read."""
    return image.dtype == np.uint8 and image.ndim == _CHANNELS and image.shape[2] == _CHANNELS


def _device(count: int, dtype: type) -> Buffer:
    return Buffer.empty((count,), dtype, MemoryLocation.DEVICE)


class _Scratch:
    """A device allocation which only grows, served as typed views."""

    def __init__(self: Self) -> None:
        self._buffer: Buffer | None = None

    def get(self: Self, shape: tuple[int, ...], dtype: type) -> Buffer:
        count = math.prod(shape)
        nbytes = count * np.dtype(dtype).itemsize
        if self._buffer is None or self._buffer.nbytes < nbytes:
            self._buffer = _device(nbytes, np.uint8)
        # a byte allocation viewed as dtype: Buffer.from_ptr keeps the allocation alive
        return Buffer.from_ptr(
            self._buffer.device_ptr, shape, dtype, MemoryLocation.DEVICE, owner=self._buffer
        )


@lru_cache(maxsize=1)
def cubic_table() -> np.ndarray:
    """
    Get OpenCV's fixed point INTER_CUBIC weights, shape (1024, 16) int16.

    Row ``(Y & 31) * 32 + (X & 31)`` holds the 4x4 taps of the 1/32 pixel offset
    (X, Y), built the way ``initInterTab2D`` does in imgwarp.cpp, float32 step by step.
    """
    f32 = np.float32
    a = f32(-0.75)
    x = np.arange(_TAB_SIZE, dtype=f32) * f32(1 / _TAB_SIZE)
    xp = x + f32(1)
    y = f32(1) - x
    c0 = ((a * xp - f32(5) * a) * xp + f32(8) * a) * xp - f32(4) * a
    c1 = ((a + f32(2)) * x - (a + f32(3))) * x * x + f32(1)
    c2 = ((a + f32(2)) * y - (a + f32(3))) * y * y + f32(1)
    c3 = f32(1) - c0 - c1 - c2
    taps = np.stack([c0, c1, c2, c3], axis=1)  # (32, 4)
    scale = f32(1 << 15)
    # (i, j, k1, k2): vertical offset i, horizontal offset j, tap k1 down, tap k2 across
    weights = taps[:, None, :, None] * taps[None, :, None, :]
    fixed = np.clip(np.rint(weights * scale), -32768, 32767).astype(np.int16)
    fixed = fixed.reshape(_TAB_SIZE * _TAB_SIZE, 16)
    for cell in fixed:
        diff = int(cell.astype(np.int64).sum()) - (1 << 15)
        if diff == 0:
            continue
        # the largest (or smallest) of the four central taps absorbs the rounding error
        big = small = 10
        for k1 in (2, 3):
            for k2 in (2, 3):
                k = k1 * 4 + k2
                if cell[k] < cell[small]:
                    small = k
                elif cell[k] > cell[big]:
                    big = k
        k = big if diff < 0 else small
        cell[k] = np.int16(np.int64(cell[k]) - diff)
    return np.ascontiguousarray(fixed)


class DeviceFrame:
    """Uploads host images into reused device buffers, one per slot."""

    def __init__(self: Self) -> None:
        self._scratch: dict[int, _Scratch] = {}
        self._stage: Buffer | None = None

    def __call__(
        self: Self, image: np.ndarray, stream: cudart.cudaStream_t, slot: int = 0
    ) -> Buffer:
        """Enqueue the upload of a (H, W, 3) uint8 image, the Buffer is valid on ``stream``."""
        frame = self._scratch.setdefault(slot, _Scratch()).get(image.shape, np.uint8)
        if self._stage is None or self._stage.nbytes < image.nbytes:
            self._stage = Buffer.empty((image.nbytes,), np.uint8, pinned=True)
        memcpy_host_to_device_staged_async(frame.device_ptr, image, self._stage.array, stream)
        return frame


class CudaTextDetectionInput:
    """Resize and normalize images into a text detector's input tensor on the device."""

    def __init__(
        self: Self,
        limit_side_len: int,
        limit_type: str,
        max_shape: tuple[int, int],
        mean: tuple[float, float, float] | None,
        std: tuple[float, float, float] | None,
        input_range: tuple[float, float],
    ) -> None:
        file, _ = OCR_PRE
        self._kernel = Kernel(file, "ocrDetResize")
        self._limit_side_len = limit_side_len
        self._limit_type = limit_type
        self._max_shape = max_shape
        lut = np.ascontiguousarray(text_detection_lut(mean, std, input_range).T)
        self._lut = Buffer.from_array(lut, MemoryLocation.DEVICE)
        self._tensor = _Scratch()
        self._frame = DeviceFrame()

    def __call__(
        self: Self,
        images: Sequence[np.ndarray | Buffer],
        stream: cudart.cudaStream_t,
    ) -> tuple[Buffer, list[tuple[float, float]], list[tuple[float, float]]]:
        """Return the (N, 3, H, W) tensor, and the ratios and padding of preprocess_text_detection."""
        sizes = [
            text_detection_size(
                img.shape[0], img.shape[1], self._limit_side_len, self._limit_type, self._max_shape
            )
            for img in images
        ]
        height = max(h for h, _ in sizes)
        width = max(w for _, w in sizes)
        tensor = self._tensor.get((len(images), 3, height, width), np.float32)
        if any(size != (height, width) for size in sizes):
            cuda_call(cudart.cudaMemsetAsync(tensor.device_ptr, 0, tensor.nbytes, stream))
        ratios: list[tuple[float, float]] = []
        padding: list[tuple[float, float]] = []
        for i, (img, (h, w)) in enumerate(zip(images, sizes)):
            if isinstance(img, Buffer):
                frame = img if img.is_device else self._frame(img.array, stream)
            else:
                frame = self._frame(img, stream)
            sh, sw = img.shape[:2]
            args = cached_args(
                self._kernel,
                frame.device_ptr,
                sh,
                sw,
                h,
                w,
                self._lut.device_ptr,
                height,
                width,
                i,
                tensor.device_ptr,
            )
            self._kernel.call((math.ceil(h * w / _THREADS), 1, 1), (_THREADS, 1, 1), stream, args)
            ratios.append((w / sw, h / sh))
            padding.append((float(width - w), float(height - h)))
        return tensor, ratios, padding


@dataclass
class CropPlan:
    """Text box crops on the device, grouped into recognizer batches."""

    order: np.ndarray  # box indices by increasing crop aspect ratio
    batches: list[tuple[int, int, int]]  # (start, stop, padded width) into order
    crops: Buffer  # the crops, back to back
    info: Buffer  # per box in order: crop offset, rows, columns, resized width


class CudaTextCrops:
    """Crop text boxes from a device frame and build text recognizer batches on the device."""

    def __init__(
        self: Self,
        image_height: int,
        max_width: int,
        batch_size: int,
        mean: tuple[float, float, float] | None,
        std: tuple[float, float, float] | None,
        input_range: tuple[float, float],
    ) -> None:
        file, _ = OCR_PRE
        self._warp = Kernel(file, "ocrWarpCrops")
        self._input = Kernel(file, "ocrRecInput")
        self._height = image_height
        self._max_width = max_width
        self._batch_size = batch_size
        self._lut = Buffer.from_array(
            text_recognition_lut(mean, std, input_range), MemoryLocation.DEVICE
        )
        self._tab = Buffer.from_array(cubic_table(), MemoryLocation.DEVICE)
        self._crops = _Scratch()
        self._mats = _Scratch()
        self._geom = _Scratch()
        self._info = _Scratch()
        self._tensor = _Scratch()

    @staticmethod
    def _upload(scratch: _Scratch, array: np.ndarray, stream: cudart.cudaStream_t) -> Buffer:
        buffer = scratch.get(array.shape, array.dtype.type)
        buffer.copy_from(Buffer.wrap(array), stream)
        return buffer

    def plan(
        self: Self,
        frame: Buffer,
        boxes: np.ndarray,
        stream: cudart.cudaStream_t,
    ) -> CropPlan:
        """
        Enqueue the perspective crops of ``boxes`` (N, 4, 2) from a (H, W, 3) uint8 device frame.

        The crops are exactly :func:`crop_text_region`'s, batched the way
        :meth:`TextRecognizer.end2end` batches them.

        Raises
        ------
        ValueError
            If a box is smaller than a pixel.

        """
        n = len(boxes)
        mats = np.empty((n, 9), np.float64)
        geom = np.empty((n, 4), np.int32)
        shapes = []  # (rows, columns) of each crop after the rotation
        biggest = 0
        offset = 0
        for i, box in enumerate(boxes):
            matrix, width, height = text_crop_geometry(box)
            if width < 1 or height < 1:
                err_msg = f"Cannot crop a text box of size {width}x{height}"
                raise ValueError(err_msg)
            # warpPerspective inverts the matrix itself, the kernel is handed the result
            mats[i] = cv2.invert(matrix)[1].reshape(9)
            rot = height / width >= _ROTATE_ASPECT
            geom[i] = (width, height, rot, offset)
            shapes.append((width, height) if rot else (height, width))
            offset += width * height * 3
            biggest = max(biggest, width * height)

        order = np.argsort([cols / rows for rows, cols in shapes])
        info = np.empty((n, 4), np.int32)
        batches = []
        for start in range(0, n, self._batch_size):
            idx = order[start : start + self._batch_size]
            width, widths = text_recognition_widths(
                [shapes[i] for i in idx], self._height, max_width=self._max_width
            )
            batches.append((start, start + len(idx), width))
            for k, (i, rw) in enumerate(zip(idx, widths)):
                info[start + k] = (geom[i, 3], *shapes[i], rw)

        crops = self._crops.get((offset,), np.uint8)
        mats_dev = self._upload(self._mats, mats, stream)
        geom_dev = self._upload(self._geom, geom, stream)
        info_dev = self._upload(self._info, info, stream)
        args = cached_args(
            self._warp,
            frame.device_ptr,
            frame.shape[0],
            frame.shape[1],
            mats_dev.device_ptr,
            geom_dev.device_ptr,
            self._tab.device_ptr,
            crops.device_ptr,
        )
        self._warp.call((math.ceil(biggest / _THREADS), n, 1), (_THREADS, 1, 1), stream, args)
        return CropPlan(order, batches, crops, info_dev)

    def batch_input(
        self: Self,
        plan: CropPlan,
        index: int,
        stream: cudart.cudaStream_t,
    ) -> Buffer:
        """Enqueue the (N, 3, H, W) float32 recognizer input of batch ``index`` of the plan."""
        start, stop, width = plan.batches[index]
        count = stop - start
        tensor = self._tensor.get((count, 3, self._height, width), np.float32)
        args = cached_args(
            self._input,
            plan.crops.device_ptr,
            plan.info.device_ptr,
            start,
            self._height,
            width,
            self._lut.device_ptr,
            tensor.device_ptr,
        )
        blocks = math.ceil(self._height * width / _THREADS)
        self._input.call((blocks, count, 1), (_THREADS, 1, 1), stream, args)
        return tensor
