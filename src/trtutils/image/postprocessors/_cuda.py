# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
CUDA postprocessors for TensorRT engine outputs.

:class:`CUDAPostprocessor`
    Postprocesses raw engine outputs on the GPU. Reads the engine's device
    output buffers directly (no D2H of the raw outputs), runs the
    postprocessing kernels on the engine stream, and copies only the
    compacted results back to the host. Returns the same per-image unified
    format as the CPU postprocessor functions in this package.

Note:
----
All returned arrays are views into pinned host buffers owned by the
postprocessor. They WILL BE OVERWRITTEN INPLACE by the next postprocess call
(same contract as ``no_copy=True`` in the CPU postprocessors).

"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import numpy as np

from trtutils._log import LOG
from trtutils.compat._libs import cudart
from trtutils.core import (
    Kernel,
    create_binding,
    memcpy_device_to_host_async,
    memcpy_host_to_device_async,
    stream_synchronize,
)
from trtutils.core._cuda import cuda_call
from trtutils.image.kernels import (
    COMPACT_BOXES,
    COMPACT_RTDETRV3,
    COMPACT_V10,
    DEPTH_MINMAX,
    DEPTH_NORMALIZE,
    HAND_NMS,
    RFDETR_GATHER,
    RFDETR_SIGMOID,
    RFDETR_TOPK,
    SOFTMAX_ROWS,
)

if TYPE_CHECKING:
    from trtutils.core._bindings import Binding


# dtype expectations per family and output index: 'f' = float32, 'i|f' = int32 or float32
# CUDA hand NMS shared-memory budget: one candidate per slot (k <= 1024)
_HAND_NMS_MAX_K = 1024

_EXPECTED_DTYPES: dict[str, tuple[str, ...]] = {
    "efficient_nms": ("i|f", "f", "f", "i|f"),
    "yolov10": ("f",),
    "detr": ("f", "i|f", "f"),
    "rtdetrv3": ("f", "i|f"),
    "rfdetr": ("f", "f"),
    "classification": ("f",),
    "depth": ("f",),
    "hand": ("f", "f", "i|f", "f", "i"),
}


class CUDAPostprocessor:
    """
    Postprocess TensorRT engine outputs on the GPU.

    Parameters
    ----------
    output_spec : list[tuple[list[int], np.dtype]]
        The engine output specs, ``(shape, dtype)``, in engine output order.
    family : str
        The postprocessor family. One of: 'efficient_nms', 'yolov10', 'detr',
        'rtdetrv3', 'rfdetr', 'classification', 'depth', 'hand'.
    k : int
        Max candidates per image (detection families). For 'rfdetr' this is
        the number of queries.
    num_classes : int, optional
        Number of classes for 'rfdetr' (from the labels output).
    total_dets : int, optional
        Total combined detection rows for 'rtdetrv3' (T in (T, 6)).
    stream : cudart.cudaStream_t
        The CUDA stream to run postprocessing kernels on. Should be the
        engine stream so execution order is preserved.
    tag : str, optional
        Tag for logging.
    pagelocked_mem : bool, optional
        Whether to allocate host buffers as pagelocked. Default is True.
    unified_mem : bool, optional
        Whether the system has unified memory.
    verbose : bool, optional
        Log extra debug information.

    Raises
    ------
    ValueError
        If a required output tensor has an unsupported dtype (float32
        expected; int32 allowed for count/class tensors).

    """

    FAMILIES = (
        "efficient_nms",
        "yolov10",
        "detr",
        "rtdetrv3",
        "rfdetr",
        "classification",
        "depth",
        "hand",
    )

    def __init__(
        self: Any,
        output_spec: list[tuple[list[int], np.dtype]],
        family: str,
        k: int,
        *,
        num_classes: int | None = None,
        total_dets: int | None = None,
        stream: cudart.cudaStream_t,
        tag: str | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        if family not in self.FAMILIES:
            err_msg = f"Invalid postprocessor family {family!r}, options are: {self.FAMILIES}"
            raise ValueError(err_msg)
        self._output_spec = output_spec
        self._family = family
        self._k = k
        self._num_classes = num_classes
        self._total_dets = total_dets
        self._stream = stream
        self._tag = tag if tag is not None else "CUDAPostprocessor"
        self._verbose = verbose if verbose is not None else False
        self._pagelocked = pagelocked_mem if pagelocked_mem is not None else True
        self._unified_mem = unified_mem
        self._buffers: dict[str, Binding] = {}
        self._kernels: dict[str, Kernel] = {}
        self._batch: int = 0
        self._pair_classes: int = self._output_spec[3][0][3] if family == "hand" else 0
        self._per_image_shape: tuple[int, ...] | None = (
            (1, *self._output_spec[0][0][1:]) if family in ("classification", "depth") else None
        )
        self._validate_dtypes()

    def _validate_dtypes(self: Any) -> None:
        expected = _EXPECTED_DTYPES[self._family]
        for idx, spec in enumerate(expected):
            if idx >= len(self._output_spec):
                continue  # optional trailing output (e.g. hand side)
            dtype = np.dtype(self._output_spec[idx][1])
            kind = dtype.kind
            ok = (
                (spec == "i|f" and kind in ("i", "f") and dtype.itemsize == 4)  # noqa: PLR2004
                or (spec == "i" and kind == "i" and dtype.itemsize == 4)  # noqa: PLR2004
                or (spec == "f" and kind == "f" and dtype.itemsize == 4)  # noqa: PLR2004
            )
            if not ok:
                err_msg = (
                    f"{self._tag}: CUDA postprocessing family '{self._family}' expects"
                    f" output {idx} dtype {spec}, found {dtype}"
                )
                raise ValueError(err_msg)

    # -- internal helpers ----------------------------------------------------

    def _is_float(self: Any, idx: int) -> int:
        return int(np.dtype(self._output_spec[idx][1]).kind == "f")

    def _buf(self: Any, key: str, batch: int, shape: tuple[int, ...], dtype: np.dtype) -> Binding:
        if self._batch != batch:
            self._free_buffers()
            self._batch = batch
        if key not in self._buffers:
            self._buffers[key] = create_binding(
                np.zeros(shape, dtype=dtype),
                name=f"{self._tag}.{key}",
                pagelocked_mem=self._pagelocked,
                unified_mem=self._unified_mem,
            )
        return self._buffers[key]

    def _free_buffers(self: Any) -> None:
        for buf in self._buffers.values():
            buf.free()
        self._buffers = {}

    def _kernel(self: Any, key: str, spec: tuple) -> Kernel:
        if key not in self._kernels:
            self._kernels[key] = Kernel(spec[0], spec[1], max_arg_cache=2)
        return self._kernels[key]

    def _upload_ratios_pads(
        self: Any,
        batch: int,
        ratios: list[tuple[float, float]],
        padding: list[tuple[float, float]],
    ) -> tuple[Binding, Binding]:
        ratios_b = self._buf("ratios", batch, (batch, 2), np.float32)
        pads_b = self._buf("pads", batch, (batch, 2), np.float32)
        memcpy_host_to_device_async(
            ratios_b.allocation,
            np.asarray(ratios, dtype=np.float32),
            self._stream,
        )
        memcpy_host_to_device_async(
            pads_b.allocation,
            np.asarray(padding, dtype=np.float32),
            self._stream,
        )
        return ratios_b, pads_b

    def _compact_common(self: Any, batch: int, k: int) -> tuple[Binding, Binding, Binding, Binding]:
        """Allocate the compact output + counts buffers and zero counts."""
        boxes_b = self._buf("boxes", batch, (batch, k, 4), np.float32)
        scores_b = self._buf("scores", batch, (batch, k), np.float32)
        classes_b = self._buf("classes", batch, (batch, k), np.int32)
        counts_b = self._buf("counts", batch, (batch,), np.int32)
        cuda_call(cudart.cudaMemsetAsync(counts_b.allocation, 0, batch * 4, self._stream))
        return boxes_b, scores_b, classes_b, counts_b

    def _finalize_dets(self: Any, batch: int) -> list[list[np.ndarray]]:
        """Copy compacted detection results to host and slice per image."""
        stream = self._stream
        counts_b = self._buffers["counts"]
        boxes_b = self._buffers["boxes"]
        scores_b = self._buffers["scores"]
        classes_b = self._buffers["classes"]
        memcpy_device_to_host_async(counts_b.host_allocation, counts_b.allocation, stream)
        memcpy_device_to_host_async(boxes_b.host_allocation, boxes_b.allocation, stream)
        memcpy_device_to_host_async(scores_b.host_allocation, scores_b.allocation, stream)
        memcpy_device_to_host_async(classes_b.host_allocation, classes_b.allocation, stream)
        stream_synchronize(stream)

        counts = counts_b.host_allocation
        results: list[list[np.ndarray]] = []
        for b in range(batch):
            n = int(counts[b])
            results.append(
                [
                    boxes_b.host_allocation[b : b + 1, :n].reshape(n, 4),
                    scores_b.host_allocation[b : b + 1, :n].reshape(n),
                    classes_b.host_allocation[b : b + 1, :n].reshape(n),
                ]
            )
        return results

    # -- detection families --------------------------------------------------

    def postprocess_efficient_nms(
        self: Any,
        batch: int,
        outputs: list[int],
        ratios: list[tuple[float, float]],
        padding: list[tuple[float, float]],
        conf_thres: float | None = None,
        *,
        verbose: bool | None = None,
    ) -> list[list[np.ndarray]]:
        """
        GPU version of :func:`postprocess_efficient_nms`.

        Parameters
        ----------
        batch : int
            The number of images in the engine outputs.
        outputs : list[int]
            Device pointers to the engine outputs
            [num_dets (B,), bboxes (B,K,4), scores (B,K), classes (B,K)].
        ratios : list[tuple[float, float]]
            Preprocessing resize ratios per image.
        padding : list[tuple[float, float]]
            Preprocessing padding per image.
        conf_thres : float, optional
            Optional extra confidence filter.

        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[list[np.ndarray]]
            One list per image: [bboxes (N,4), scores (N,), class_ids (N,)].

        """
        if verbose:
            LOG.debug(f"{self._tag}: cuda postprocess (efficient_nms), batch: {batch}")
        num_dets_ptr, boxes_ptr, scores_ptr, classes_ptr = outputs
        k = self._k
        ratios_b, pads_b = self._upload_ratios_pads(batch, ratios, padding)
        boxes_b, scores_b, classes_b, counts_b = self._compact_common(batch, k)

        kernel = self._kernel("compact", COMPACT_BOXES)
        args = kernel.create_args(
            boxes_ptr,
            scores_ptr,
            classes_ptr,
            ratios_b.allocation,
            pads_b.allocation,
            num_dets_ptr,
            boxes_b.allocation,
            scores_b.allocation,
            classes_b.allocation,
            counts_b.allocation,
            0,  # out_indices (unused)
            batch,
            k,
            float(conf_thres) if conf_thres is not None else 0.0,
            int(conf_thres is not None),
            0,  # use_finite
            1,  # use_rescale
            self._is_float(3),
            self._is_float(0),
        )
        kernel.call(((batch * k + 255) // 256, 1, 1), (256, 1, 1), self._stream, args)
        return self._finalize_dets(batch)

    def postprocess_yolov10(
        self: Any,
        batch: int,
        outputs: list[int],
        ratios: list[tuple[float, float]],
        padding: list[tuple[float, float]],
        conf_thres: float | None = None,
        *,
        verbose: bool | None = None,
    ) -> list[list[np.ndarray]]:
        """
        GPU version of :func:`postprocess_yolov10`.

        Parameters
        ----------
        batch : int
            The number of images in the engine outputs.
        outputs : list[int]
            Device pointers to the engine outputs: [output (B, N, 6)].
        ratios : list[tuple[float, float]]
            Preprocessing resize ratios per image.
        padding : list[tuple[float, float]]
            Preprocessing padding per image.
        conf_thres : float, optional
            Optional extra confidence filter.

        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[list[np.ndarray]]
            One list per image: [bboxes (N,4), scores (N,), class_ids (N,)].

        """
        if verbose:
            LOG.debug(f"{self._tag}: cuda postprocess (yolov10), batch: {batch}")
        v10_ptr = outputs[0]
        k = self._k
        ratios_b, pads_b = self._upload_ratios_pads(batch, ratios, padding)
        boxes_b, scores_b, classes_b, counts_b = self._compact_common(batch, k)

        kernel = self._kernel("compact_v10", COMPACT_V10)
        args = kernel.create_args(
            v10_ptr,
            ratios_b.allocation,
            pads_b.allocation,
            boxes_b.allocation,
            scores_b.allocation,
            classes_b.allocation,
            counts_b.allocation,
            batch,
            k,
            float(conf_thres) if conf_thres is not None else 0.0,
            int(conf_thres is not None),
        )
        kernel.call(((batch * k + 255) // 256, 1, 1), (256, 1, 1), self._stream, args)
        return self._finalize_dets(batch)

    def postprocess_detr(
        self: Any,
        batch: int,
        outputs: list[int],
        ratios: list[tuple[float, float]],  # noqa: ARG002
        padding: list[tuple[float, float]],  # noqa: ARG002
        conf_thres: float | None = None,
        *,
        verbose: bool | None = None,
    ) -> list[list[np.ndarray]]:
        """
        GPU version of :func:`postprocess_detr`.

        Parameters
        ----------
        batch : int
            The number of images in the engine outputs.
        outputs : list[int]
            Device pointers in (scores, labels, boxes) order:
            [scores (B,Q), labels (B,Q), boxes (B,Q,4)].
            Boxes are already in original image coordinates; no rescale.
        ratios : list[tuple[float, float]]
            Unused.
        padding : list[tuple[float, float]]
            Unused.
        conf_thres : float, optional
            Confidence threshold to filter detections.

        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[list[np.ndarray]]
            One list per image: [bboxes (N,4), scores (N,), class_ids (N,)].

        """
        if verbose:
            LOG.debug(f"{self._tag}: cuda postprocess (detr), batch: {batch}")
        scores_ptr, labels_ptr, boxes_ptr = outputs
        k = self._k
        boxes_b, scores_b, classes_b, counts_b = self._compact_common(batch, k)

        kernel = self._kernel("compact", COMPACT_BOXES)
        args = kernel.create_args(
            boxes_ptr,
            scores_ptr,
            labels_ptr,
            0,  # ratios (unused)
            0,  # pads (unused)
            0,  # num_dets (unused)
            boxes_b.allocation,
            scores_b.allocation,
            classes_b.allocation,
            counts_b.allocation,
            0,  # out_indices (unused)
            batch,
            k,
            float(conf_thres) if conf_thres is not None else 0.0,
            int(conf_thres is not None),
            1,  # use_finite
            0,  # use_rescale
            self._is_float(1),
            0,
        )
        kernel.call(((batch * k + 255) // 256, 1, 1), (256, 1, 1), self._stream, args)
        return self._finalize_dets(batch)

    def postprocess_rtdetrv3(
        self: Any,
        batch: int,
        outputs: list[int],
        ratios: list[tuple[float, float]],  # noqa: ARG002
        padding: list[tuple[float, float]],  # noqa: ARG002
        conf_thres: float | None = None,
        *,
        verbose: bool | None = None,
    ) -> list[list[np.ndarray]]:
        """
        GPU version of :func:`postprocess_rtdetrv3`.

        Parameters
        ----------
        batch : int
            The number of images in the engine outputs.
        outputs : list[int]
            Device pointers: [combined_dets (T,6), num_dets_per_image (B,)].
        ratios : list[tuple[float, float]]
            Unused (boxes already in image coords).
        padding : list[tuple[float, float]]
            Unused.
        conf_thres : float, optional
            Confidence threshold to filter detections.

        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[list[np.ndarray]]
            One list per image: [bboxes (N,4), scores (N,), class_ids (N,)].

        """
        if verbose:
            LOG.debug(f"{self._tag}: cuda postprocess (rtdetrv3), batch: {batch}")
        dets_ptr, num_dets_ptr = outputs
        total = self._total_dets or self._k
        k = self._k
        boxes_b, scores_b, classes_b, counts_b = self._compact_common(batch, k)

        kernel = self._kernel("compact_rtdetrv3", COMPACT_RTDETRV3)
        args = kernel.create_args(
            dets_ptr,
            num_dets_ptr,
            boxes_b.allocation,
            scores_b.allocation,
            classes_b.allocation,
            counts_b.allocation,
            batch,
            total,
            k,
            float(conf_thres) if conf_thres is not None else 0.0,
            int(conf_thres is not None),
            self._is_float(1),
        )
        kernel.call(((total + 255) // 256, 1, 1), (256, 1, 1), self._stream, args)
        return self._finalize_dets(batch)

    def postprocess_rfdetr(
        self: Any,
        batch: int,
        outputs: list[int],
        ratios: list[tuple[float, float]],
        padding: list[tuple[float, float]],
        conf_thres: float | None = None,
        input_size: tuple[int, int] | None = None,
        *,
        verbose: bool | None = None,
    ) -> list[list[np.ndarray]]:
        """
        GPU version of :func:`postprocess_rfdetr`.

        Parameters
        ----------
        batch : int
            The number of images in the engine outputs.
        outputs : list[int]
            Device pointers: [dets (B,Q,4), labels (B,Q,C)] with dets
            normalized (cx, cy, w, h) and labels as logits.
        ratios : list[tuple[float, float]]
            Preprocessing resize ratios per image.
        padding : list[tuple[float, float]]
            Preprocessing padding per image.
        conf_thres : float, optional
            Confidence threshold to filter detections.
        input_size : tuple[int, int], optional
            Model input (width, height) to denormalize boxes. Default 640.

        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[list[np.ndarray]]
            One list per image: [bboxes (N,4), scores (N,), class_ids (N,)].

        """
        if verbose:
            LOG.debug(f"{self._tag}: cuda postprocess (rfdetr), batch: {batch}")
        dets_ptr, logits_ptr = outputs
        q = self._k
        c = self._num_classes or self._output_spec[1][0][2]
        total = batch * q * c
        ratios_b, pads_b = self._upload_ratios_pads(batch, ratios, padding)
        boxes_b, scores_b, classes_b, counts_b = self._compact_common(batch, q)
        probs_b = self._buf("probs", batch, (batch, q * c), np.float32)
        topk_s_b = self._buf("topk_scores", batch, (batch, q), np.float32)
        topk_i_b = self._buf("topk_idx", batch, (batch, q), np.int32)

        stream = self._stream
        sig = self._kernel("rfdetr_sig", RFDETR_SIGMOID)
        sig.call(
            ((total + 255) // 256, 1, 1),
            (256, 1, 1),
            stream,
            sig.create_args(logits_ptr, probs_b.allocation, total),
        )
        topk = self._kernel("rfdetr_topk", RFDETR_TOPK)
        topk.call(
            (batch, 1, 1),
            (1024, 1, 1),
            stream,
            topk.create_args(probs_b.allocation, topk_s_b.allocation, topk_i_b.allocation, q, q * c),
        )
        input_w, input_h = input_size if input_size is not None else (640, 640)
        gather = self._kernel("rfdetr_gather", RFDETR_GATHER)
        gather.call(
            ((batch * q + 255) // 256, 1, 1),
            (256, 1, 1),
            stream,
            gather.create_args(
                dets_ptr,
                topk_s_b.allocation,
                topk_i_b.allocation,
                ratios_b.allocation,
                pads_b.allocation,
                boxes_b.allocation,
                scores_b.allocation,
                classes_b.allocation,
                counts_b.allocation,
                q,
                c,
                float(input_w),
                float(input_h),
                float(conf_thres) if conf_thres is not None else 0.0,
                int(conf_thres is not None),
            ),
        )
        return self._finalize_dets(batch)

    # -- classification / depth ----------------------------------------------

    def postprocess_classifications(
        self: Any,
        batch: int,
        outputs: list[int],
        *,
        verbose: bool | None = None,
    ) -> list[list[np.ndarray]]:
        """
        GPU version of :func:`postprocess_classifications`.

        Parameters
        ----------
        batch : int
            The number of images in the engine outputs.
        outputs : list[int]
            Device pointers: [logits (B, ...)].

        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[list[np.ndarray]]
            Per-image probability arrays with the batch dim stripped.

        """
        if verbose:
            LOG.debug(f"{self._tag}: cuda postprocess (classification), batch: {batch}")
        logits_ptr = outputs[0]
        c = int(np.prod(self._output_spec[0][0][1:]))
        probs_b = self._buf("probs", batch, (batch, c), np.float32)

        kernel = self._kernel("softmax", SOFTMAX_ROWS)
        kernel.call(
            (batch, 1, 1),
            (1024, 1, 1),
            self._stream,
            kernel.create_args(logits_ptr, probs_b.allocation, c),
        )
        memcpy_device_to_host_async(probs_b.host_allocation, probs_b.allocation, self._stream)
        stream_synchronize(self._stream)

        shape = self._per_image_shape or (1, c)
        return [[probs_b.host_allocation[b : b + 1].reshape(shape)] for b in range(batch)]

    def postprocess_depth(
        self: Any,
        batch: int,
        outputs: list[int],
        *,
        verbose: bool | None = None,
    ) -> list[list[np.ndarray]]:
        """
        GPU version of :func:`postprocess_depth`.

        Parameters
        ----------
        batch : int
            The number of images in the engine outputs.
        outputs : list[int]
            Device pointers: [depth (B, H, W) or (B, 1, H, W)] float32.

        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[list[np.ndarray]]
            Per-image depth maps of shape (1, H, W) normalized to [0, 1].

        """
        if verbose:
            LOG.debug(f"{self._tag}: cuda postprocess (depth), batch: {batch}")
        depth_ptr = outputs[0]
        hwp = int(np.prod(self._output_spec[0][0][1:]))
        minmax_b = self._buf("minmax", batch, (batch, 2), np.float32)
        depth_b = self._buf("depth", batch, (batch, hwp), np.float32)

        stream = self._stream
        mm = self._kernel("depth_minmax", DEPTH_MINMAX)
        mm.call(
            (batch, 1, 1),
            (1024, 1, 1),
            stream,
            mm.create_args(depth_ptr, minmax_b.allocation, hwp),
        )
        norm = self._kernel("depth_norm", DEPTH_NORMALIZE)
        norm.call(
            ((batch * hwp + 255) // 256, 1, 1),
            (256, 1, 1),
            stream,
            norm.create_args(depth_ptr, minmax_b.allocation, batch, hwp),
        )
        memcpy_device_to_host_async(depth_b.host_allocation, depth_ptr, stream)
        stream_synchronize(stream)

        shape = self._per_image_shape or (1, hwp)
        return [[depth_b.host_allocation[b : b + 1].reshape(shape)] for b in range(batch)]

    # -- hand interaction ------------------------------------------------------

    def postprocess_hand_interactions(
        self: Any,
        batch: int,
        outputs: list[int],
        ratios: list[tuple[float, float]],
        padding: list[tuple[float, float]],
        conf_thres: float,
        nms_iou_thres: float,
        *,
        verbose: bool | None = None,
    ) -> list[list[np.ndarray]]:
        """
        GPU version of :func:`postprocess_hand_interactions`.

        Parameters
        ----------
        batch : int
            The number of images in the engine outputs.
        outputs : list[int]
            Device pointers: [boxes (B,K,4), scores (B,K), labels (B,K),
            pair_probs (B,K,K,C), (side (B,K))].
        ratios : list[tuple[float, float]]
            Preprocessing resize ratios per image.
        padding : list[tuple[float, float]]
            Preprocessing padding per image.
        conf_thres : float
            Confidence threshold used to filter candidates.
        nms_iou_thres : float
            IoU threshold for class-aware NMS.

        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[list[np.ndarray]]
            One list per image: [bboxes (N,4), scores (N,), labels (N,),
            pair_probs (N,N,C), side (N,1) or (N,0)].

        """
        if verbose:
            LOG.debug(f"{self._tag}: cuda postprocess (hand), batch: {batch}")
        boxes_ptr, scores_ptr, labels_ptr, pairs_ptr = outputs[:4]
        has_side = len(outputs) == 5  # noqa: PLR2004
        side_ptr = outputs[4] if has_side else 0
        k = self._k
        if k > _HAND_NMS_MAX_K:
            err_msg = f"{self._tag}: CUDA hand NMS requires K <= {_HAND_NMS_MAX_K}, found {k}"
            raise ValueError(err_msg)

        ratios_b, pads_b = self._upload_ratios_pads(batch, ratios, padding)

        # stage 1: remap + conf filter (same kernel as the other detectors);
        # the index map (compacted slot -> original candidate) is needed by stage 2
        # because pair_probs/side live in the original candidate space
        idx_b = self._buf("compact_idx", batch, (batch, k), np.int32)
        boxes_b, scores_b, classes_b, counts_b = self._compact_common(batch, k)
        compact = self._kernel("compact", COMPACT_BOXES)
        compact_args = compact.create_args(
            boxes_ptr,
            scores_ptr,
            labels_ptr,
            ratios_b.allocation,
            pads_b.allocation,
            0,  # num_dets (unused)
            boxes_b.allocation,
            scores_b.allocation,
            classes_b.allocation,
            counts_b.allocation,
            idx_b.allocation,
            batch,
            k,
            float(conf_thres),
            1,  # use_conf
            0,  # use_finite
            1,  # use_rescale
            self._is_float(2),
            0,
        )
        compact.call(((batch * k + 255) // 256, 1, 1), (256, 1, 1), self._stream, compact_args)

        # stage 2: class-aware NMS + pair gather
        c = self._pair_classes
        nms_out_boxes_b = self._buf("nms_boxes", batch, (batch, k, 4), np.float32)
        nms_out_scores_b = self._buf("nms_scores", batch, (batch, k), np.float32)
        nms_out_labels_b = self._buf("nms_labels", batch, (batch, k), np.int32)
        nms_out_pairs_b = self._buf("nms_pairs", batch, (batch, k, k, c), np.float32)
        nms_out_side_b = self._buf("nms_side", batch, (batch, k), np.int32)

        nms = self._kernel("hand_nms", HAND_NMS)
        nms_args = nms.create_args(
            boxes_b.allocation,
            scores_b.allocation,
            classes_b.allocation,
            pairs_ptr,
            side_ptr,
            idx_b.allocation,
            nms_out_boxes_b.allocation,
            nms_out_scores_b.allocation,
            nms_out_labels_b.allocation,
            nms_out_pairs_b.allocation,
            nms_out_side_b.allocation,
            counts_b.allocation,
            k,
            c,
            float(nms_iou_thres),
            float(conf_thres),
            int(has_side),
        )
        nms.call((batch, 1, 1), (1024, 1, 1), self._stream, nms_args)

        # copy results to host
        stream = self._stream
        memcpy_device_to_host_async(counts_b.host_allocation, counts_b.allocation, stream)
        memcpy_device_to_host_async(
            nms_out_boxes_b.host_allocation, nms_out_boxes_b.allocation, stream
        )
        memcpy_device_to_host_async(
            nms_out_scores_b.host_allocation, nms_out_scores_b.allocation, stream
        )
        memcpy_device_to_host_async(
            nms_out_labels_b.host_allocation, nms_out_labels_b.allocation, stream
        )
        memcpy_device_to_host_async(
            nms_out_pairs_b.host_allocation, nms_out_pairs_b.allocation, stream
        )
        if has_side:
            memcpy_device_to_host_async(
                nms_out_side_b.host_allocation, nms_out_side_b.allocation, stream
            )
        stream_synchronize(stream)

        counts = counts_b.host_allocation
        results: list[list[np.ndarray]] = []
        for b in range(batch):
            n = int(counts[b])
            pairs = nms_out_pairs_b.host_allocation[b].reshape(k, k, c)[:n, :n, :]
            side = (
                nms_out_side_b.host_allocation[b : b + 1, :n].reshape(n, 1)
                if has_side
                else np.zeros((n, 0), dtype=np.int32)
            )
            results.append(
                [
                    nms_out_boxes_b.host_allocation[b : b + 1, :n].reshape(n, 4),
                    nms_out_scores_b.host_allocation[b : b + 1, :n].reshape(n),
                    nms_out_labels_b.host_allocation[b : b + 1, :n].reshape(n),
                    pairs,
                    side,
                ]
            )
        return results

    # -- lifecycle -------------------------------------------------------------

    def free(self: Any) -> None:
        """Free all allocated device/host buffers and kernel modules."""
        self._free_buffers()
        for kernel in self._kernels.values():
            kernel.free()
        self._kernels = {}

    def __del__(self: Any) -> None:
        with contextlib.suppress(AttributeError, RuntimeError, SystemError):
            self.free()
