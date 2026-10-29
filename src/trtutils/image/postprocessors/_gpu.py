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
from trtutils.core._kernels import Kernel, cached_args, f64
from trtutils.core._memory import memcpy_device_to_host_async, memcpy_host_to_device_async
from trtutils.core._stream import stream_synchronize
from trtutils.image.kernels import YOLO_POST

from ._obb import _probiou_reach

if TYPE_CHECKING:
    from typing_extensions import Self

    from ._segmentation import MaskCanvas

# column stripes per row block of the NMS mask kernel
_NMS_STRIPES = 16
_RANK_SLICES = 16  # RANK_SLICES in yolo_post.cu
# NMS_MAX_WORDS in yolo_post.cu
_MAX_CANDIDATES = 64 * 1024
# NMS bit matrix rows before a frame needs more: (N, N / 64) words is 141 MB at N = 33600
_NMS_ROWS = 4096


def _device(shape: tuple[int, ...], dtype: type) -> Buffer:
    return Buffer.empty(shape, dtype, MemoryLocation.DEVICE)


def _mapped(shape: tuple[int, ...], dtype: type) -> Buffer:
    # pinned host memory the kernels write directly, read after a stream sync
    return Buffer.empty(shape, dtype, MemoryLocation.HOST, pinned=True, mapped=True)


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
        self._scatter = Kernel(file, "yoloScatter")
        self._mask = Kernel(file, "yoloNmsMask")
        self._reduce = Kernel(file, "yoloNmsReduce")
        self._shape: tuple[int, int, int, int] | None = None

    def _allocate(self: Self, batch: int, num: int, max_det: int) -> None:
        self.count = _device((batch,), np.int32)
        self._cand_anchor = _device((batch, num), np.int32)
        self._cand_score = _device((batch, num), np.float32)
        self._cand_cls = _device((batch, num), np.int32)
        self._cand_rank = _device((batch, num), np.int32)
        self.sort_anchor = _device((batch, num), np.int32)
        self.sort_score = _device((batch, num), np.float32)
        self.sort_cls = _device((batch, num), np.int32)
        self.sort_box = _device((batch, num, 4), np.float32)
        self._bits: Buffer | None = None
        self._rows = min(num, _NMS_ROWS)
        self.keep = _device((batch, max_det), np.int32)
        self.keep_count = _device((batch,), np.int32)

    def rank(
        self: Self,
        head: Buffer,
        cls_start: int,
        nc: int,
        conf_thres: float | None,
        max_det: int,
        stream: cudart.cudaStream_t,
        *,
        inclusive: bool = False,
    ) -> None:
        """
        Enqueue the decode and sort of ``head`` on ``stream``, filling count and the sorted arrays.

        Keeps scores above ``conf_thres`` (at or above with ``inclusive``); None keeps every anchor.
        """
        batch, channels, num = head.shape
        if self._shape != (batch, channels, num, max_det):
            self._allocate(batch, num, max_det)
            self._shape = (batch, channels, num, max_det)
        unfiltered = -math.inf if inclusive else 0.0
        conf = unfiltered if conf_thres is None else float(conf_thres)
        cuda_call(cudart.cudaMemsetAsync(self.count.ptr, 0, self.count.nbytes, stream))
        args = cached_args(
            self._decode,
            head.device_ptr,
            channels,
            num,
            cls_start,
            nc,
            conf,
            int(inclusive),
            self.count.ptr,
            self._cand_anchor.ptr,
            self._cand_score.ptr,
            self._cand_cls.ptr,
            self._cand_rank.ptr,
        )
        self._decode.call((math.ceil(num / 256), batch, 1), (256, 1, 1), stream, args)
        args = cached_args(
            self._rank,
            num,
            self.count.ptr,
            self._cand_anchor.ptr,
            self._cand_score.ptr,
            self._cand_rank.ptr,
        )
        blocks = math.ceil(num / 256)
        self._rank.call((blocks * _RANK_SLICES, batch, 1), (256, 1, 1), stream, args)
        args = cached_args(
            self._scatter,
            head.device_ptr,
            channels,
            num,
            self.count.ptr,
            self._cand_anchor.ptr,
            self._cand_score.ptr,
            self._cand_cls.ptr,
            self._cand_rank.ptr,
            self.sort_anchor.ptr,
            self.sort_score.ptr,
            self.sort_cls.ptr,
            self.sort_box.ptr,
        )
        self._scatter.call((blocks, batch, 1), (256, 1, 1), stream, args)

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
        """
        Enqueue NMS of ``head`` on ``stream``, filling keep / keep_count and the sorted arrays.

        keep_count is -1 for an image with more candidates than the NMS holds: call
        :meth:`grow` and run the frame again.
        """
        batch, _, num = head.shape
        if num > _MAX_CANDIDATES:
            err_msg = f"CUDA NMS supports up to {_MAX_CANDIDATES} anchors, got {num}"
            raise ValueError(err_msg)
        self.rank(head, cls_start, nc, conf_thres, max_det, stream)
        rows = self._rows
        words = math.ceil(rows / 64)
        if self._bits is None:
            self._bits = _device((batch, rows, words), np.uint64)
        args = cached_args(
            self._mask,
            rows,
            self.count.ptr,
            self.sort_cls.ptr,
            self.sort_box.ptr,
            float(nms_iou_thres),
            self._bits.ptr,
        )
        self._mask.call((words, _NMS_STRIPES, batch), (64, 1, 1), stream, args)
        args = cached_args(
            self._reduce,
            rows,
            max_det,
            self.count.ptr,
            self._bits.ptr,
            self.keep.ptr,
            self.keep_count.ptr,
        )
        self._reduce.call((batch, 1, 1), (32, 1, 1), stream, args)

    def grow(self: Self) -> None:
        """Size the NMS for every anchor, after a frame overflowed it."""
        self._rows = self.sort_anchor.shape[1]
        self._bits = None


class CudaObbPostprocessor:
    """YOLO oriented box postprocessing on the device, see ``postprocess_yolo_obb``."""

    def __init__(self: Self) -> None:
        file, _ = YOLO_POST
        self._nms = YoloNms()
        self._prep = Kernel(file, "obbPrep")
        self._suppress = Kernel(file, "obbSuppress")
        self._keep = Kernel(file, "obbKeep")
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
        stream: cudart.cudaStream_t,
    ) -> list[list[np.ndarray]]:
        """Postprocess device ``[output0]`` into the per-image lists of ``postprocess_yolo_obb``."""
        head = outputs[0]
        batch, channels, num = head.shape
        if self._shape != (batch, num, max_det):
            self._boxes = _device((batch, num, 8), np.float32)
            self._alive = _device((batch, num), np.uint8)
            self._dets = _mapped((batch, max_det, 11), np.float32)
            self._keep_count = _mapped((batch,), np.int32)
            self._letterbox = _mapped((batch, 4), np.float32)
            self._shape = (batch, num, max_det)

        for i in range(batch):
            if not np.isclose(ratios[i][0], ratios[i][1]):
                err_msg = "OBB decoding needs uniform scaling, use resize_method='letterbox'"
                raise ValueError(err_msg)
            self._letterbox.array[i] = (*ratios[i], *padding[i])

        nms = self._nms
        nms.rank(head, 4, channels - 5, conf_thres, max_det, stream, inclusive=True)
        args = cached_args(
            self._prep,
            head.device_ptr,
            channels,
            num,
            nms.count.ptr,
            nms.sort_anchor.ptr,
            self._letterbox.device_ptr,
            self._boxes.ptr,
        )
        self._prep.call((math.ceil(num / 256), batch, 1), (256, 1, 1), stream, args)
        args = cached_args(
            self._suppress,
            num,
            nms.count.ptr,
            nms.sort_cls.ptr,
            self._boxes.ptr,
            float(nms_iou_thres),
            _probiou_reach(nms_iou_thres),
            self._alive.ptr,
        )
        # one warp per box
        self._suppress.call((math.ceil(num * 32 / 256), batch, 1), (256, 1, 1), stream, args)
        args = cached_args(
            self._keep,
            num,
            max_det,
            nms.count.ptr,
            self._alive.ptr,
            nms.sort_cls.ptr,
            nms.sort_score.ptr,
            self._boxes.ptr,
            self._dets.device_ptr,
            self._keep_count.device_ptr,
        )
        self._keep.call((batch, 1, 1), (1024, 1, 1), stream, args)
        stream_synchronize(stream)

        results = []
        for i in range(batch):
            dets = self._dets.array[i, : int(self._keep_count.array[i])]
            results.append(
                [
                    dets[:, :4].copy(),
                    dets[:, 4].copy(),
                    dets[:, 5].astype(np.int64),
                    dets[:, 6:].copy(),
                ],
            )
        return results


class CudaSegPostprocessor:
    """YOLO segmentation postprocessing on the device, see ``postprocess_yolo_seg``."""

    def __init__(self: Self) -> None:
        file, _ = YOLO_POST
        self._nms = YoloNms()
        self._gather = Kernel(file, "yoloSegGather")
        self._probs_kernel = Kernel(file, "yoloSegProbs")
        self._masks = Kernel(file, "yoloSegMasks")
        self._clear = Kernel(file, "yoloSegClear")
        self._shape: tuple[int, ...] | None = None

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
        _, nm, mh, mw = proto.shape
        if self._shape != (batch, nm, mh, mw, max_det):
            # the mask kernels read dets and stale boxes from device memory: reads of mapped
            # memory each cost a PCIe round trip
            self._dets = _device((batch, max_det, 6), np.float32)
            self._dets_host = _mapped((batch, max_det, 6), np.float32)
            self._coefs = _device((batch, max_det, nm), np.float32)
            # boxes each mask wrote this frame, and the previous frame's to clear
            self._regions = _mapped((batch, max_det, 4), np.int32)
            self._stale = _mapped((batch, max_det, 4), np.int32)
            self._stale_dev = _device((batch, max_det, 4), np.int32)
            self._keep_count = _mapped((batch,), np.int32)
            self._letterbox = _mapped((batch, 6), np.float32)
            # per-detection sigmoid probs of the proto cells, reused image by image
            self._probs = _device((max_det, mh * mw), np.float32)
            self._shape = (batch, nm, mh, mw, max_det)

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

        def draw(i: int, n: int) -> np.ndarray:
            width, height = sizes[i]
            view = canvases[i].take(n, height, width, clear=False)
            count_ptr = nms.keep_count.ptr + i * 4
            self._draw_masks(
                proto,
                i,
                n,
                count_ptr,
                sizes[i],
                input_size,
                padding[i],
                mask_thres,
                canvases[i],
                stream,
            )
            return view

        args = cached_args(
            self._gather,
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
            self._dets.ptr,
            self._coefs.ptr,
            self._keep_count.device_ptr,
        )
        self._gather.call((max_det, batch, 1), (32, 1, 1), stream, args)
        memcpy_device_to_host_async(self._dets_host.array, self._dets.ptr, stream)

        # draw into each canvas's current depth before the counts are known, so one sync
        # covers the usual frame; the kernels stop at the kept count on the device
        caps = [canvases[i].capacity(sizes[i][1], sizes[i][0]) for i in range(batch)]
        masks = [draw(i, caps[i]) for i in range(batch)]
        stream_synchronize(stream)

        # more masks than the canvas holds (or a new canvas): grow it and draw again
        counts = [int(c) for c in self._keep_count.array]
        if min(counts) < 0:
            nms.grow()
            return self(
                outputs,
                ratios,
                padding,
                conf_thres,
                nms_iou_thres,
                mask_thres,
                max_det=max_det,
                input_size=input_size,
                canvases=canvases,
                stream=stream,
            )
        redo = [i for i in range(batch) if counts[i] > caps[i]]
        for i in redo:
            masks[i] = draw(i, counts[i])
        if redo:
            stream_synchronize(stream)

        results = []
        for i in range(batch):
            n = counts[i]
            regions = self._regions.array[i, :n].tolist()
            masks[i] = masks[i][:n]
            canvases[i].written = [
                (k, by1, by2, bx1, bx2) for k, (bx1, by1, bx2, by2) in enumerate(regions)
            ]
            dets = self._dets_host.array[i, :n]
            results.append(
                [dets[:, :4].copy(), dets[:, 4].copy(), dets[:, 5].astype(np.int64), masks[i]],
            )
        return results

    def _draw_masks(
        self: Self,
        proto: Buffer,
        index: int,
        n: int,
        count_ptr: int,
        size: tuple[int, int],
        input_size: tuple[int, int],
        padding: tuple[float, float],
        mask_thres: float,
        canvas: MaskCanvas,
        stream: cudart.cudaStream_t,
    ) -> None:
        """Enqueue clearing the canvas's previous boxes and drawing image ``index``'s n masks."""
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
        geometry = (
            width,
            height,
            f64(scale_x),
            f64(scale_y),
            f64(x0 + 0.5 * scale_x - 0.5),
            f64(y0 + 0.5 * scale_y - 0.5),
        )
        dets = self._dets.ptr + index * self._dets.shape[1] * 6 * 4

        # zero the previous frame's boxes on the device instead of the CPU
        if canvas.written:
            stale_n = max(entry[0] for entry in canvas.written) + 1
            stale = self._stale.array[index]
            stale[:stale_n] = 0
            for idx, by1, by2, bx1, bx2 in canvas.written:
                stale[idx] = (bx1, by1, bx2, by2)
            stale_ptr = self._stale_dev.ptr + index * self._stale.shape[1] * 4 * 4
            memcpy_host_to_device_async(stale_ptr, stale[:stale_n], stream)
            args = cached_args(
                self._clear,
                canvas.device_ptr,
                stale_ptr,
                stale_n,
                width,
                height,
                dets,
                n,
                count_ptr,
                mh,
                mw,
                *geometry[2:],
            )
            self._clear.call((stale_n, 16, 1), (256, 1, 1), stream, args)
        if n == 0:
            return

        args = cached_args(
            self._probs_kernel,
            proto.device_ptr + index * nm * mh * mw * 4,
            nm,
            mh,
            mw,
            dets,
            self._coefs.ptr + index * self._coefs.shape[1] * nm * 4,
            n,
            count_ptr,
            *geometry,
            self._probs.ptr,
        )
        self._probs_kernel.call((n, 4, 1), (256, 1, 1), stream, args)
        args = cached_args(
            self._masks,
            self._probs.ptr,
            mh,
            mw,
            dets,
            n,
            count_ptr,
            *geometry,
            float(mask_thres),
            canvas.device_ptr,
            self._regions.device_ptr + index * self._regions.shape[1] * 4 * 4,
        )
        self._masks.call((n, 32, 1), (256, 1, 1), stream, args)


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
        args = cached_args(
            self._gather,
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
            self._keep_count.device_ptr,
        )
        self._gather.call((max_det, batch, 1), (32, 1, 1), stream, args)
        stream_synchronize(stream)
        if self._keep_count.array.min() < 0:
            nms.grow()
            return self(
                outputs,
                ratios,
                padding,
                conf_thres,
                nms_iou_thres,
                max_det=max_det,
                input_size=input_size,
                stream=stream,
            )

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
