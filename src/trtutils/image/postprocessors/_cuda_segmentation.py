# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""CUDA postprocessing for YOLO segmentation, run on the engine stream."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from trtutils.core._kernels import f64
from trtutils.core._memory import memcpy_device_to_host_async, memcpy_host_to_device_async
from trtutils.image.kernels import YOLO_SEG

from trtutils.core._buffer import Buffer

from ._cuda import CUDAPostprocessor, split_detections
from ._cuda_nms import YoloNms
from ._cpu_segmentation import _mask_geometry

if TYPE_CHECKING:
    from typing_extensions import Self

    from trtutils.compat._libs import cudart

    from ._cpu_segmentation import MaskCanvas

_FLOAT = 4  # bytes, also int32


class CUDASegPostprocessor(CUDAPostprocessor):
    """YOLO segmentation postprocessing on the device, see ``postprocess_yolo_seg``."""

    _FILE = YOLO_SEG

    def __init__(self: Self, stream: cudart.cudaStream_t) -> None:
        super().__init__(stream)
        self._nms = YoloNms(stream)
        self._probs_kernel = self._kernel("yoloSegProbs")
        self._masks = self._kernel("yoloSegMasks")
        self._clear = self._kernel("yoloSegClear")
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
    ) -> list[list[np.ndarray]]:
        """
        Postprocess device outputs ``[output0, output1]`` into one mapped canvas per image.

        Returns the same per-image lists as ``postprocess_yolo_seg``; the masks
        are views of the canvases.
        """
        head, proto = outputs
        batch, channels, _ = head.shape
        _, nm, mh, mw = proto.shape
        if self._shape != (batch, nm, mh, mw, max_det):
            # the mask kernels read dets and stale boxes from device memory: reads of mapped
            # memory each cost a PCIe round trip
            self._dets = Buffer.device((batch, max_det, 6), np.float32)
            self._dets_host = Buffer.host_mapped((batch, max_det, 6), np.float32)
            self._coefs = Buffer.device((batch, max_det, nm), np.float32)
            self._counts = Buffer.host_mapped((batch,), np.int32)
            # boxes each mask wrote this frame, and the previous frame's to clear
            self._regions = Buffer.host_mapped((batch, max_det, 4), np.int32)
            self._stale = Buffer.host_mapped((batch, max_det, 4), np.int32)
            self._stale_dev = Buffer.device((batch, max_det, 4), np.int32)
            # per-detection sigmoid probs of the proto cells, reused image by image
            self._probs = Buffer.device((max_det, mh * mw), np.float32)
            self._shape = (batch, nm, mh, mw, max_det)

        geoms = [_mask_geometry(input_size, r, p, (mh, mw)) for r, p in zip(ratios, padding)]
        self._nms(head, 4, channels - 4 - nm, conf_thres, nms_iou_thres, max_det)
        self._nms.gather(
            head,
            self._letterbox(ratios, padding, input_size),
            self._dets.ptr,
            self._counts.device_ptr,
            extra_start=channels - nm,
            extras=(self._coefs.ptr, nm),
        )
        memcpy_device_to_host_async(self._dets_host.array, self._dets.ptr, self._stream)

        def draw(i: int, n: int) -> np.ndarray:
            return self._draw(proto, i, n, geoms[i], mask_thres, canvases[i])

        # draw into each canvas's current depth before the counts are known, so one sync
        # covers the usual frame; the kernels stop at the kept count on the device
        caps = [canvas.capacity(g[1], g[0]) for canvas, g in zip(canvases, geoms)]
        masks = [draw(i, n) for i, n in enumerate(caps)]
        self._sync()

        # more masks than the canvas holds (or a new canvas): grow it and draw again
        counts = self._counts.array.tolist()
        redo = [i for i in range(batch) if counts[i] > caps[i]]
        for i in redo:
            masks[i] = draw(i, counts[i])
        if redo:
            self._sync()

        results = split_detections(self._dets_host.array, self._counts.array)
        for i, (result, canvas) in enumerate(zip(results, canvases)):
            n = counts[i]
            regions = self._regions.array[i, :n].tolist()
            canvas.written = [
                (k, by1, by2, bx1, bx2) for k, (bx1, by1, bx2, by2) in enumerate(regions)
            ]
            result.append(masks[i][:n])
        return results

    def _draw(
        self: Self,
        proto: Buffer,
        index: int,
        n: int,
        geometry: tuple[int, int, float, float, float, float],
        mask_thres: float,
        canvas: MaskCanvas,
    ) -> np.ndarray:
        """Enqueue clearing the canvas's previous boxes and drawing image ``index``'s n masks."""
        _, nm, mh, mw = proto.shape
        max_det = self._dets.shape[1]
        width, height, *scales = geometry
        view = canvas.take(n, height, width, clear=False)
        geom = (width, height, *(f64(s) for s in scales))
        dets = self._dets.ptr + index * max_det * 6 * _FLOAT
        count = self._nms.keep_count.ptr + index * _FLOAT

        # zero the previous frame's boxes on the device instead of the CPU
        if canvas.written:
            stale_n = max(entry[0] for entry in canvas.written) + 1
            stale = self._stale.array[index]
            stale[:stale_n] = 0
            for k, by1, by2, bx1, bx2 in canvas.written:
                stale[k] = (bx1, by1, bx2, by2)
            stale_ptr = self._stale_dev.ptr + index * max_det * 4 * _FLOAT
            memcpy_host_to_device_async(stale_ptr, stale[:stale_n], self._stream)
            self._launch(
                self._clear,
                (stale_n, 16, 1),
                (256, 1, 1),
                canvas.device_ptr,
                stale_ptr,
                stale_n,
                width,
                height,
                dets,
                n,
                count,
                mh,
                mw,
                *geom[2:],
            )
        if n == 0:
            return view

        self._launch(
            self._probs_kernel,
            (n, 4, 1),
            (256, 1, 1),
            proto.device_ptr + index * nm * mh * mw * _FLOAT,
            nm,
            mh,
            mw,
            dets,
            self._coefs.ptr + index * max_det * nm * _FLOAT,
            n,
            count,
            *geom,
            self._probs.ptr,
        )
        self._launch(
            self._masks,
            (n, 32, 1),
            (256, 1, 1),
            self._probs.ptr,
            mh,
            mw,
            dets,
            n,
            count,
            *geom,
            float(mask_thres),
            canvas.device_ptr,
            self._regions.device_ptr + index * max_det * 4 * _FLOAT,
        )
        return view
