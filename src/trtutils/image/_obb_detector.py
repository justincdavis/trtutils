# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Oriented bounding box (OBB) detector implementation."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, overload

import numpy as np
import nvtx
from typing_extensions import TypeGuard

from trtutils._flags import FLAGS
from trtutils._log import LOG

from ._detector import Detector
from ._schema import OutputSchema
from .interfaces import OBBDetectorInterface
from .postprocessors._obb import get_obb_detections, postprocess_yolo_obb
from .preprocessors import CUDAPreprocessor, TRTPreprocessor

if TYPE_CHECKING:
    from pathlib import Path

    from typing_extensions import Self

    from ._schema import InputSchema
    from .postprocessors._obb import OBBDetection


def _is_postprocessed_outputs(
    outputs: list[np.ndarray] | list[list[np.ndarray]],
) -> TypeGuard[list[list[np.ndarray]]]:
    return not outputs or isinstance(outputs[0], list)


class OBBDetector(Detector, OBBDetectorInterface):
    """
    Implementation of oriented bounding box (OBB) detectors.

    Wraps raw ultralytics OBB task heads (no EfficientNMS grafted). Postprocessed
    per-image outputs are ``[bboxes (N,4), scores (N,), class_ids (N,), rboxes
    (N,5)]``; ``rboxes`` is the authoritative ``(cx, cy, w, h, angle)`` with angle
    in radians, ``bboxes`` is the derived enclosing axis-aligned box so the
    inherited :meth:`get_detections` keeps working unchanged.
    """

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0.0, 1.0),
        preprocessor: str = "trt",
        resize_method: str = "letterbox",
        conf_thres: float = 0.1,
        nms_iou_thres: float = 0.5,
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
        input_schema: InputSchema | str | None = None,
        dla_core: int | None = None,
        device: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = None,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        """
        Create an OBBDetector object.

        Parameters
        ----------
        engine_path : Path, str
            The Path or str to the compiled TensorRT engine.
        warmup_iterations : int
            The number of warmup iterations to perform.
            The default is 10.
        input_range : tuple[float, float]
            The range of input values which should be passed to
            the model. By default [0.0, 1.0].
        preprocessor : str
            The type of preprocessor to use.
            The options are ['cpu', 'cuda', 'trt'], default is 'trt'.
        resize_method : str
            The type of resize algorithm to use.
            The options are ['letterbox', 'linear'], default is 'letterbox'.
        conf_thres : float
            The confidence threshold above which to generate detections.
            By default 0.1
        nms_iou_thres : float
            The IOU threshold used for the per-class rotated NMS operation.
            By default, 0.5
        mean : tuple[float, float, float] | None, optional
            The mean values to use for the imagenet normalization.
            By default, None, which means no normalization will be applied.
        std : tuple[float, float, float] | None, optional
            The standard deviation values to use for the imagenet normalization.
            By default, None, which means no normalization will be applied.
        input_schema : InputSchema, str, optional
            Manually specify the input schema instead of auto-detection.
            By default None, which means the schema will be auto-detected from
            the engine's input names.
        dla_core : int, optional
            The DLA core to assign DLA layers of the engine to. Default is None.
            If None, any DLA layers will be assigned to DLA core 0.
        device : int, optional
            The CUDA device index to use for this detector. Default is None,
            which uses the current device.
        backend : str
            The execution backend to use. Options are ['auto', 'async_v3', 'async_v2'].
            Default is 'auto', which selects the best available backend.
        warmup : bool, optional
            Whether or not to perform warmup iterations.
        pagelocked_mem : bool, optional
            Whether or not to use pagelocked memory for underlying CUDA operations.
            By default, pagelocked memory will be used.
        unified_mem : bool, optional
            Whether or not the system has unified memory.
            If True, use cudaHostAllocMapped to take advantage of unified memory.
            By default None, which means the default host allocation will be used.
        cuda_graph : bool, optional
            Whether or not to enable CUDA graph capture for optimized execution.
            Only effective with async_v3 backend. Default is True.
        extra_nms : bool, optional
            Whether or not an additional CPU-side NMS operation should be
            conducted by the inherited :meth:`get_detections`. Has no effect
            on :meth:`get_obb_detections`, whose rotated NMS is configured via
            ``nms_iou_thres`` above.
        agnostic_nms : bool, optional
            Whether or not the optional/additional NMS operation for
            :meth:`get_detections` should perform class agnostic NMS.
        no_warn : bool, optional
            If True, suppresses warnings from TensorRT during engine deserialization.
            Default is None, which means warnings will be shown.
        verbose : bool, optional
            Whether or not to log additional information.
            Only covers the initialization phase.

        """
        Detector.__init__(
            self,
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_range=input_range,
            preprocessor=preprocessor,
            resize_method=resize_method,
            conf_thres=conf_thres,
            nms_iou_thres=nms_iou_thres,
            mean=mean,
            std=std,
            input_schema=input_schema,
            output_schema=OutputSchema.YOLO_OBB,
            dla_core=dla_core,
            device=device,
            backend=backend,
            warmup=warmup,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
            cuda_graph=cuda_graph,
            extra_nms=extra_nms,
            agnostic_nms=agnostic_nms,
            no_warn=no_warn,
            verbose=verbose,
        )

        # rebind to inject nms_iou_thres, which Detector.postprocess's call site does not pass
        self._postprocess_fn = partial(postprocess_yolo_obb, nms_iou_thres=nms_iou_thres)

        # prepend with 'obb_' to avoid conflicts with ImageModel/Detector._nvtx_tags
        self._nvtx_tags.update(
            {
                "obb_get_obb_detections": f"obb_detector::get_obb_detections [{self._tag}]",
                "obb_end2end": f"obb_detector::end2end [{self._tag}]",
                "obb__end2end": f"obb_detector::_end2end [{self._tag}]",
                "obb__end2end_graph": f"obb_detector::_end2end_graph [{self._tag}]",
            }
        )

    # get_obb_detections overloads
    @overload
    def get_obb_detections(
        self: Self,
        outputs: list[np.ndarray],
        conf_thres: float | None = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[OBBDetection]: ...

    @overload
    def get_obb_detections(
        self: Self,
        outputs: list[list[np.ndarray]],
        conf_thres: float | None = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[list[OBBDetection]]: ...

    def get_obb_detections(
        self: Self,
        outputs: list[np.ndarray] | list[list[np.ndarray]],
        conf_thres: float | None = None,
        *,
        verbose: bool | None = None,
    ) -> list[OBBDetection] | list[list[OBBDetection]]:
        """
        Get the oriented bounding box detections from postprocessed outputs.

        Parameters
        ----------
        outputs : list[np.ndarray] | list[list[np.ndarray]]
            For single image: list[np.ndarray] (single image's postprocessed outputs).
            For batch: list[list[np.ndarray]] (postprocessed outputs per image).
        conf_thres : float, optional
            The confidence threshold with which to retrieve detections.
            By default None, which will use value passed during initialization.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[OBBDetection] | list[list[OBBDetection]]
            For single image: list[OBBDetection] (detections for single image).
            For batch: list[list[OBBDetection]] (detections per image).

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["obb_get_obb_detections"])

        if verbose:
            LOG.debug(f"{self._tag}: get_obb_detections")

        is_single = outputs and isinstance(outputs[0], np.ndarray)
        conf_thres = conf_thres if conf_thres is not None else self._conf_thres

        if is_single:
            batch_outputs: list[list[np.ndarray]] = [outputs]  # ty: ignore[invalid-assignment]
            result = get_obb_detections(batch_outputs, conf_thres, verbose=verbose)
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()  # get_obb_detections
            return result[0]

        result_batch = get_obb_detections(
            outputs,  # ty: ignore[invalid-argument-type]
            conf_thres,
            verbose=verbose,
        )

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # get_obb_detections

        return result_batch

    # end2end overloads
    @overload
    def end2end(
        self: Self,
        images: np.ndarray,
        conf_thres: float | None = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[OBBDetection]: ...

    @overload
    def end2end(
        self: Self,
        images: list[np.ndarray],
        conf_thres: float | None = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[list[OBBDetection]]: ...

    def end2end(  # ty: ignore[invalid-method-override]
        self: Self,
        images: np.ndarray | list[np.ndarray],
        conf_thres: float | None = None,
        *,
        verbose: bool | None = None,
    ) -> list[OBBDetection] | list[list[OBBDetection]]:
        """
        Perform end to end inference for a batch of images.

        Equivalent to running preprocess, run, postprocess, and
        get_obb_detections in that order. Makes some memory transfer
        optimizations under the hood to improve performance.

        Parameters
        ----------
        images : np.ndarray | list[np.ndarray]
            A single image (HWC format) or list of images to perform inference with.
        conf_thres : float, optional
            The confidence threshold with which to retrieve detections.
            By default None, which uses the value provided during initialization.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[OBBDetection] | list[list[OBBDetection]]
            For single image: list[OBBDetection] (detections).
            For batch: list[list[OBBDetection]] (detections per image).

        Raises
        ------
        RuntimeError
            If end2end_graph is enabled and image dimensions change after first call.
        RuntimeError
            If end2end_graph is enabled and CUDA graph capture fails.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["obb_end2end"])

        if verbose:
            LOG.debug(f"{self._tag}: end2end")

        if isinstance(images, np.ndarray):
            batch_images: list[np.ndarray] = [images]
            is_single = True
        else:
            batch_images = images
            is_single = False

        if self._e2e_graph_enabled:
            result = self._end2end_graph(batch_images, conf_thres=conf_thres, verbose=verbose)
        else:
            result = self._end2end(batch_images, conf_thres=conf_thres, verbose=verbose)

        if is_single:
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()  # end2end
            return result[0]

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # end2end

        return result

    def _end2end(  # ty: ignore[invalid-method-override]
        self: Self,
        images: list[np.ndarray],
        conf_thres: float | None = None,
        *,
        verbose: bool | None = None,
    ) -> list[list[OBBDetection]]:
        """Execute the standard end2end path without graph capture."""
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["obb__end2end"])

        outputs: list[np.ndarray] | list[list[np.ndarray]]
        postprocessed: list[list[np.ndarray]]
        if not isinstance(self._preprocessor, (CUDAPreprocessor, TRTPreprocessor)):
            if verbose:
                LOG.debug(f"{self._tag}: end2end -> calling CPU preprocess")

            outputs = self.run(
                images,
                conf_thres=conf_thres,
                preprocessed=False,
                postprocess=True,
                no_copy=True,
                verbose=verbose,
            )
            if not _is_postprocessed_outputs(outputs):
                err_msg = "Expected postprocessed detector outputs in end2end."
                if FLAGS.NVTX_ENABLED:
                    nvtx.pop_range()  # _end2end
                raise RuntimeError(err_msg)
            postprocessed = outputs
        else:
            if verbose:
                LOG.debug(f"{self._tag}: end2end -> calling CUDA preprocess")

            gpu_ptr, ratios, padding = self._preprocessor.direct_preproc(
                images,
                resize=self._resize_method,
                no_warn=True,
                verbose=verbose,
            )
            raw_outputs = self._engine.direct_exec([gpu_ptr], no_warn=True)
            postprocessed = self.postprocess(
                raw_outputs,
                ratios,
                padding,
                conf_thres,
                no_copy=True,
                verbose=verbose,
            )

        result = self.get_obb_detections(postprocessed, conf_thres, verbose=verbose)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # _end2end

        return result

    def _end2end_graph(  # ty: ignore[invalid-method-override]
        self: Self,
        images: list[np.ndarray],
        conf_thres: float | None = None,
        *,
        verbose: bool | None = None,
    ) -> list[list[OBBDetection]]:
        """
        Execute graph-accelerated end2end path.

        This implementation captures only TRTEngine inference in the CUDA graph.
        Preprocessing runs outside the graph since H2D copies cannot be captured.
        Supports CPU, CUDA, and TRT preprocessors.
        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["obb__end2end_graph"])

        raw_outputs, ratios, padding = self._end2end_graph_core(images, verbose=verbose)

        postprocessed: list[list[np.ndarray]] = self.postprocess(
            raw_outputs,
            ratios,
            padding,
            conf_thres,
            no_copy=True,
            verbose=verbose,
        )
        result = self.get_obb_detections(postprocessed, conf_thres, verbose=verbose)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # _end2end_graph

        return result
