# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""CUDA postprocessing for YOLO-style heads, run on the engine stream."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from trtutils.compat._libs import cudart
from trtutils.core._buffer import Buffer, MemoryLocation
from trtutils.core._cuda import cuda_call
from trtutils.core._kernels import Kernel
from trtutils.core._stream import stream_synchronize
from trtutils.image.kernels import YOLO_POST

if TYPE_CHECKING:
    from typing_extensions import Self

    from ._segmentation import MaskCanvas

# NMS_MAX_WORDS in yolo_post.cu
_MAX_CANDIDATES = 64 * 1024


def _device(shape: tuple[int, ...], dtype: type) -> Buffer:
    return Buffer.empty(shape, dtype, MemoryLocation.DEVICE)


def _mapped(shape: tuple[int, ...], dtype: type) -> Buffer:
    # pinned host memory the kernels write directly, read after a stream sync
    return Buffer.empty(shape, dtype, MemoryLocation.HOST, pinned=True, mapped=True)


def _f64(value: float) -> np.ndarray:
    return np.array([value], dtype=np.float64)


class YoloNms:
    """
    Decode, sort, and class-aware NMS of a (B, C, N) YOLO head on the device.

    Matches the CPU path: candidates scoring above ``conf_thres`` sorted by
    score (ties by anchor), then greedy NMS as cv2.dnn.NMSBoxesBatched.
    """

    def __init__(self: Self) -> None:
        file, _ = YOLO_POST
        self._decode = Kernel(file, "yoloDecode")
        self._rank = Kernel(file, "yoloRank")
        self._mask = Kernel(file, "yoloNmsMask")
        self._reduce = Kernel(file, "yoloNmsReduce")
        self._shape: tuple[int, int, int, int] | None = None

    def _allocate(self: Self, batch: int, num: int, max_det: int) -> None:
        if num > _MAX_CANDIDATES:
            err_msg = f"CUDA NMS supports up to {_MAX_CANDIDATES} anchors, got {num}"
            raise ValueError(err_msg)
        words = math.ceil(num / 64)
        self.count = _device((batch,), np.int32)
        self._cand_anchor = _device((batch, num), np.int32)
        self._cand_score = _device((batch, num), np.float32)
        self._cand_cls = _device((batch, num), np.int32)
        self.sort_anchor = _device((batch, num), np.int32)
        self.sort_score = _device((batch, num), np.float32)
        self.sort_cls = _device((batch, num), np.int32)
        self.sort_box = _device((batch, num, 4), np.float32)
        self._bits = _device((batch, num, words), np.uint64)
        self.keep = _device((batch, max_det), np.int32)
        self.keep_count = _device((batch,), np.int32)

    def __call__(
        self: Self,
        head: Buffer,
        cls_start: int,
        nc: int,
        conf_thres: float | None,
        nms_iou_thres: float,
        max_det: int,
        stream: cudart.cudaStream_t,
    ) -> None:
        """Enqueue NMS of ``head`` on ``stream``, filling keep / keep_count and the sorted arrays."""
        batch, channels, num = head.shape
        if self._shape != (batch, channels, num, max_det):
            self._allocate(batch, num, max_det)
            self._shape = (batch, channels, num, max_det)
        conf = 0.0 if conf_thres is None else float(conf_thres)
        words = math.ceil(num / 64)
        cuda_call(cudart.cudaMemsetAsync(self.count.ptr, 0, self.count.nbytes, stream))
        args = self._decode.create_args(
            head.device_ptr,
            channels,
            num,
            cls_start,
            nc,
            conf,
            self.count.ptr,
            self._cand_anchor.ptr,
            self._cand_score.ptr,
            self._cand_cls.ptr,
        )
        self._decode.call((math.ceil(num / 256), batch, 1), (256, 1, 1), stream, args)
        args = self._rank.create_args(
            head.device_ptr,
            channels,
            num,
            self.count.ptr,
            self._cand_anchor.ptr,
            self._cand_score.ptr,
            self._cand_cls.ptr,
            self.sort_anchor.ptr,
            self.sort_score.ptr,
            self.sort_cls.ptr,
            self.sort_box.ptr,
        )
        self._rank.call((math.ceil(num / 256), batch, 1), (256, 1, 1), stream, args)
        args = self._mask.create_args(
            num,
            self.count.ptr,
            self.sort_cls.ptr,
            self.sort_box.ptr,
            float(nms_iou_thres),
            self._bits.ptr,
        )
        self._mask.call((words, words, batch), (64, 1, 1), stream, args)
        args = self._reduce.create_args(
            num,
            max_det,
            self.count.ptr,
            self._bits.ptr,
            self.keep.ptr,
            self.keep_count.ptr,
        )
        self._reduce.call((batch, 1, 1), (128, 1, 1), stream, args)


class CudaSegPostprocessor:
    """YOLO segmentation postprocessing on the device, see ``postprocess_yolo_seg``."""

    def __init__(self: Self) -> None:
        file, _ = YOLO_POST
        self._nms = YoloNms()
        self._gather = Kernel(file, "yoloSegGather")
        self._masks = Kernel(file, "yoloSegMasks")
        self._shape: tuple[int, int, int] | None = None

    def __call__(
        self: Self,
        outputs: list[Buffer],
        ratios: list[tuple[float, float]],
        padding: list[tuple[float, float]],
        conf_thres: float | None,
        nms_iou_thres: float,
        mask_thres: float,
        *,
        max_det: int,
        input_size: tuple[int, int],
        canvases: list[MaskCanvas],
        stream: cudart.cudaStream_t,
    ) -> list[list[np.ndarray]]:
        """
        Postprocess device outputs ``[output0, output1]`` into one mapped canvas per image.

        Returns the same per-image lists as ``postprocess_yolo_seg``; the masks
        are views of the canvases.
        """
        head, proto = outputs
        batch, channels, num = head.shape
        nm = proto.shape[1]
        if self._shape != (batch, nm, max_det):
            self._dets = _mapped((batch, max_det, 6), np.float32)
            self._coefs = _device((batch, max_det, nm), np.float32)
            self._regions = _mapped((max_det, 4), np.int32)
            self._keep_count = _mapped((batch,), np.int32)
            self._letterbox = _mapped((batch, 6), np.float32)
            self._shape = (batch, nm, max_det)

        input_w, input_h = input_size
        sizes = []
        for i in range(batch):
            (rx, ry), (px, py) = ratios[i], padding[i]
            width = max(round((input_w - 2 * px) / rx), 1)
            height = max(round((input_h - 2 * py) / ry), 1)
            self._letterbox.array[i] = (rx, ry, px, py, width, height)
            sizes.append((width, height))

        self._nms(head, 4, channels - 4 - nm, conf_thres, nms_iou_thres, max_det, stream)
        nms = self._nms
        cuda_call(
            cudart.cudaMemcpyAsync(
                self._keep_count.ptr,
                nms.keep_count.ptr,
                nms.keep_count.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                stream,
            ),
        )
        args = self._gather.create_args(
            head.device_ptr,
            channels,
            num,
            channels - nm,
            nm,
            max_det,
            nms.keep.ptr,
            nms.keep_count.ptr,
            nms.sort_anchor.ptr,
            nms.sort_score.ptr,
            nms.sort_cls.ptr,
            nms.sort_box.ptr,
            self._letterbox.device_ptr,
            self._dets.device_ptr,
            self._coefs.ptr,
        )
        self._gather.call((max_det, batch, 1), (32, 1, 1), stream, args)
        stream_synchronize(stream)

        results = []
        for i in range(batch):
            n = int(self._keep_count.array[i])
            width, height = sizes[i]
            masks = canvases[i].take(n, height, width)
            if n > 0:
                self._launch_masks(
                    proto,
                    i,
                    n,
                    (width, height),
                    input_size,
                    padding[i],
                    mask_thres,
                    canvases[i],
                    stream,
                )
            dets = self._dets.array[i, :n]
            results.append(
                [dets[:, :4].copy(), dets[:, 4].copy(), dets[:, 5].astype(np.int64), masks],
            )
        return results

    def _launch_masks(
        self: Self,
        proto: Buffer,
        index: int,
        n: int,
        size: tuple[int, int],
        input_size: tuple[int, int],
        padding: tuple[float, float],
        mask_thres: float,
        canvas: MaskCanvas,
        stream: cudart.cudaStream_t,
    ) -> None:
        _, nm, mh, mw = proto.shape
        width, height = size
        input_w, input_h = input_size
        # proto-grid mapping of the letterbox content region, as _decode_masks
        stride_x = input_w / mw
        stride_y = input_h / mh
        x0 = max(int(np.floor(padding[0] / stride_x)), 0)
        y0 = max(int(np.floor(padding[1] / stride_y)), 0)
        x1 = max(min(int(np.ceil((input_w - padding[0]) / stride_x)), mw), x0 + 1)
        y1 = max(min(int(np.ceil((input_h - padding[1]) / stride_y)), mh), y0 + 1)
        scale_x = (x1 - x0) / width
        scale_y = (y1 - y0) / height
        args = self._masks.create_args(
            proto.device_ptr + index * nm * mh * mw * 4,
            nm,
            mh,
            mw,
            self._dets.device_ptr + index * self._dets.shape[1] * 6 * 4,
            self._coefs.ptr + index * self._coefs.shape[1] * nm * 4,
            n,
            width,
            height,
            _f64(scale_x),
            _f64(scale_y),
            _f64(x0 + 0.5 * scale_x - 0.5),
            _f64(y0 + 0.5 * scale_y - 0.5),
            float(mask_thres),
            canvas.device_ptr,
            self._regions.device_ptr,
        )
        self._masks.call((n, 32, 1), (256, 1, 1), stream, args)
        stream_synchronize(stream)
        regions = self._regions.array[:n].tolist()
        canvas.written = [(k, by1, by2, bx1, bx2) for k, (bx1, by1, bx2, by2) in enumerate(regions)]


class CudaPosePostprocessor:
    """YOLO pose postprocessing on the device, see ``postprocess_yolo_pose``."""

    def __init__(self: Self) -> None:
        file, _ = YOLO_POST
        self._nms = YoloNms()
        self._gather = Kernel(file, "yoloPoseGather")
        self._shape: tuple[int, int, int] | None = None

    def __call__(
        self: Self,
        outputs: list[Buffer],
        ratios: list[tuple[float, float]],
        padding: list[tuple[float, float]],
        conf_thres: float | None,
        nms_iou_thres: float,
        *,
        max_det: int,
        input_size: tuple[int, int],
        stream: cudart.cudaStream_t,
    ) -> list[list[np.ndarray]]:
        """Postprocess a device ``[output0]`` into the per-image lists of ``postprocess_yolo_pose``."""
        head = outputs[0]
        batch, channels, num = head.shape
        nk = (channels - 5) // 3
        if self._shape != (batch, nk, max_det):
            self._dets = _mapped((batch, max_det, 6), np.float32)
            self._kpts = _mapped((batch, max_det, nk, 3), np.float32)
            self._keep_count = _mapped((batch,), np.int32)
            self._letterbox = _mapped((batch, 6), np.float32)
            self._shape = (batch, nk, max_det)

        input_w, input_h = input_size
        for i in range(batch):
            (rx, ry), (px, py) = ratios[i], padding[i]
            width = max(round((input_w - 2 * px) / rx), 1)
            height = max(round((input_h - 2 * py) / ry), 1)
            self._letterbox.array[i] = (rx, ry, px, py, width, height)

        nms = self._nms
        nms(head, 4, 1, conf_thres, nms_iou_thres, max_det, stream)
        cuda_call(
            cudart.cudaMemcpyAsync(
                self._keep_count.ptr,
                nms.keep_count.ptr,
                nms.keep_count.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                stream,
            ),
        )
        args = self._gather.create_args(
            head.device_ptr,
            channels,
            num,
            5,
            nk,
            max_det,
            nms.keep.ptr,
            nms.keep_count.ptr,
            nms.sort_anchor.ptr,
            nms.sort_score.ptr,
            nms.sort_cls.ptr,
            nms.sort_box.ptr,
            self._letterbox.device_ptr,
            self._dets.device_ptr,
            self._kpts.device_ptr,
        )
        self._gather.call((max_det, batch, 1), (32, 1, 1), stream, args)
        stream_synchronize(stream)

        results = []
        for i in range(batch):
            n = int(self._keep_count.array[i])
            dets = self._dets.array[i, :n]
            results.append(
                [
                    dets[:, :4].copy(),
                    dets[:, 4].copy(),
                    dets[:, 5].astype(np.int64),
                    self._kpts.array[i, :n].copy(),
                ],
            )
        return results
