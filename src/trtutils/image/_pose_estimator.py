# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Implementation of pose estimation models."""

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
from .interfaces import PoseEstimatorInterface
from .postprocessors._pose import get_poses, postprocess_yolo_pose
from .preprocessors import CUDAPreprocessor, TRTPreprocessor

if TYPE_CHECKING:
    from pathlib import Path

    from typing_extensions import Self

    from .postprocessors._pose import Pose


def _is_postprocessed_outputs(
    outputs: list[np.ndarray] | list[list[np.ndarray]],
) -> TypeGuard[list[list[np.ndarray]]]:
    return not outputs or isinstance(outputs[0], list)


class PoseEstimator(Detector, PoseEstimatorInterface):
    """
    Implementation of pose estimation models.

    Wraps raw ultralytics pose-head engines (no EfficientNMS graft) that emit a
    single ``output0`` tensor of shape ``(batch, 5 + K*3, N)``. Postprocessed
    per-image outputs are ``[bboxes (N,4), scores (N,), class_ids (N,) all zero,
    keypoints (N,K,3)]``, so the inherited :meth:`get_detections` keeps working
    unchanged while :meth:`get_poses` zips in the keypoints.
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
        dla_core: int | None = None,
        device: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        """
        Create a PoseEstimator object.

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
            The confidence threshold above which to generate poses.
            By default 0.1.
        nms_iou_thres : float
            The IOU threshold to use during NMS. By default, 0.5.
        mean : tuple[float, float, float] | None, optional
            The mean values to use for the imagenet normalization.
            By default, None, which means no normalization will be applied.
        std : tuple[float, float, float] | None, optional
            The standard deviation values to use for the imagenet normalization.
            By default, None, which means no normalization will be applied.
        dla_core : int, optional
            The DLA core to assign DLA layers of the engine to. Default is None.
            If None, any DLA layers will be assigned to DLA core 0.
        device : int, optional
            The CUDA device index to use for this pose estimator. Default is
            None, which uses the current device.
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
            output_schema=OutputSchema.YOLO_POSE,
            dla_core=dla_core,
            device=device,
            backend=backend,
            warmup=warmup,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
            cuda_graph=cuda_graph,
            no_warn=no_warn,
            verbose=verbose,
        )

        # Detector.postprocess does not forward nms_iou_thres to _postprocess_fn,
        # so rebind here with it bound in via partial
        self._postprocess_fn = partial(postprocess_yolo_pose, nms_iou_thres=nms_iou_thres)

        # prepend with 'pose_' to avoid conflicts with ImageModel/Detector nvtx tags
        self._nvtx_tags.update(
            {
                "pose_init": f"pose_estimator::init [{self._tag}]",
                "pose_get_poses": f"pose_estimator::get_poses [{self._tag}]",
                "pose_end2end": f"pose_estimator::end2end [{self._tag}]",
                "pose__end2end": f"pose_estimator::_end2end [{self._tag}]",
                "pose__end2end_graph": f"pose_estimator::_end2end_graph [{self._tag}]",
            }
        )

        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["pose_init"])

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # init

    # get_poses overloads
    @overload
    def get_poses(
        self: Self,
        outputs: list[np.ndarray],
        conf_thres: float | None = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[Pose]: ...

    @overload
    def get_poses(
        self: Self,
        outputs: list[list[np.ndarray]],
        conf_thres: float | None = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[list[Pose]]: ...

    def get_poses(
        self: Self,
        outputs: list[np.ndarray] | list[list[np.ndarray]],
        conf_thres: float | None = None,
        *,
        verbose: bool | None = None,
    ) -> list[Pose] | list[list[Pose]]:
        """
        Get the poses from postprocessed outputs.

        Parameters
        ----------
        outputs : list[np.ndarray] | list[list[np.ndarray]]
            For single image: list[np.ndarray] (single image's postprocessed outputs).
            For batch: list[list[np.ndarray]] (postprocessed outputs per image).
        conf_thres : float, optional
            The confidence threshold with which to retrieve poses.
            By default None, which will use the value passed during initialization.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[Pose] | list[list[Pose]]
            For single image: list[Pose] (poses for single image).
            For batch: list[list[Pose]] (poses per image).

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["pose_get_poses"])

        if verbose:
            LOG.debug(f"{self._tag}: get_poses")

        is_single = outputs and isinstance(outputs[0], np.ndarray)
        conf_thres = conf_thres if conf_thres is not None else self._conf_thres

        if is_single:
            batch_outputs: list[list[np.ndarray]] = [outputs]  # ty: ignore[invalid-assignment]
            result = get_poses(batch_outputs, conf_thres, verbose=verbose)
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()  # get_poses
            return result[0]

        result_batch = get_poses(
            outputs,  # ty: ignore[invalid-argument-type]
            conf_thres,
            verbose=verbose,
        )

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # get_poses

        return result_batch

    # end2end overloads
    @overload
    def end2end(
        self: Self,
        images: np.ndarray,
        conf_thres: float | None = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[Pose]: ...

    @overload
    def end2end(
        self: Self,
        images: list[np.ndarray],
        conf_thres: float | None = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[list[Pose]]: ...

    def end2end(  # ty: ignore[invalid-method-override]
        self: Self,
        images: np.ndarray | list[np.ndarray],
        conf_thres: float | None = None,
        *,
        verbose: bool | None = None,
    ) -> list[Pose] | list[list[Pose]]:
        """
        Perform end to end inference for a batch of images.

        Equivalent to running preprocess, run, postprocess, and get_poses in
        that order. Makes some memory transfer optimizations under the hood to
        improve performance.

        Parameters
        ----------
        images : np.ndarray | list[np.ndarray]
            A single image (HWC format) or list of images to perform inference with.
        conf_thres : float, optional
            The confidence threshold with which to retrieve poses.
            By default None, which uses the value provided during initialization.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[Pose] | list[list[Pose]]
            For single image: list[Pose] (poses).
            For batch: list[list[Pose]] (poses per image).

        Raises
        ------
        RuntimeError
            If the orig_image_size buffer is not valid
        RuntimeError
            If the scale_factor buffer is not valid
        RuntimeError
            If end2end_graph is enabled and image dimensions change after first call.
        RuntimeError
            If end2end_graph is enabled and CUDA graph capture fails.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["pose_end2end"])

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
    ) -> list[list[Pose]]:
        """Execute the standard end2end path without graph capture."""
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["pose__end2end"])

        outputs: list[np.ndarray] | list[list[np.ndarray]]
        postprocessed: list[list[np.ndarray]]
        # if using CPU preprocessor best you can do is remove host-to-host copies
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
                err_msg = "Expected postprocessed pose estimator outputs in end2end."
                if FLAGS.NVTX_ENABLED:
                    nvtx.pop_range()  # _end2end
                raise RuntimeError(err_msg)
            postprocessed = outputs
        else:
            if verbose:
                LOG.debug(f"{self._tag}: end2end -> calling CUDA preprocess")

            # if using CUDA, can remove much more
            gpu_ptr, ratios, padding = self._preprocessor.direct_preproc(
                images,
                resize=self._resize_method,
                no_warn=True,
                verbose=verbose,
            )

            input_ptrs = [gpu_ptr]
            if self._use_image_size:
                orig_size_ptr, valid = self._preprocessor.orig_size_allocation
                if valid:
                    input_ptrs.append(orig_size_ptr)
                else:
                    err_msg = "orig_image_size buffer not valid"
                    if FLAGS.NVTX_ENABLED:
                        nvtx.pop_range()  # _end2end
                    raise RuntimeError(err_msg)
            if self._use_scale_factor:
                scale_ptr, scale_valid = self._preprocessor.scale_factor_allocation
                if scale_valid:
                    input_ptrs.append(scale_ptr)
                else:
                    err_msg = "scale_factor buffer not valid"
                    if FLAGS.NVTX_ENABLED:
                        nvtx.pop_range()  # _end2end
                    raise RuntimeError(err_msg)

            raw_outputs = self._engine.direct_exec(input_ptrs, no_warn=True)
            postprocessed = self.postprocess(
                raw_outputs,
                ratios,
                padding,
                conf_thres,
                no_copy=True,
                verbose=verbose,
            )

        result = self.get_poses(postprocessed, conf_thres, verbose=verbose)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # _end2end

        return result

    def _end2end_graph(  # ty: ignore[invalid-method-override]
        self: Self,
        images: list[np.ndarray],
        conf_thres: float | None = None,
        *,
        verbose: bool | None = None,
    ) -> list[list[Pose]]:
        """
        Execute graph-accelerated end2end path.

        This implementation captures only TRTEngine inference in the CUDA graph.
        Preprocessing runs outside the graph since H2D copies cannot be captured.
        Supports CPU, CUDA, and TRT preprocessors.
        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["pose__end2end_graph"])

        raw_outputs, ratios, padding = self._end2end_graph_core(images, verbose=verbose)

        postprocessed = self.postprocess(
            raw_outputs,
            ratios,
            padding,
            conf_thres,
            no_copy=True,
            verbose=verbose,
        )
        result = self.get_poses(postprocessed, conf_thres, verbose=verbose)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # _end2end_graph

        return result
