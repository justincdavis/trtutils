# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Oriented bounding box (OBB) detector implementation."""

from __future__ import annotations

import time
from functools import partial
from typing import TYPE_CHECKING, overload

import numpy as np
import nvtx
from typing_extensions import Literal, TypeGuard

from trtutils._flags import FLAGS
from trtutils._log import LOG

from ._image_model import ImageModel
from ._schema import OBBOutputSchema, resolve_obb_schemas
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


class OBBDetector(ImageModel, OBBDetectorInterface):
    """
    Implementation of oriented bounding box (OBB) detectors.

    Wraps dense prediction heads with a trailing angle channel (no NMS
    grafted). Postprocessed per-image outputs are ``[bboxes (N,4), scores
    (N,), class_ids (N,), rboxes (N,5)]``; ``rboxes`` is the authoritative
    ``(cx, cy, w, h, angle)`` with angle in radians, ``bboxes`` is the
    derived enclosing axis-aligned box.
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
        output_schema: OBBOutputSchema | str | None = None,
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
        output_schema : OBBOutputSchema, str, optional
            Manually specify the output schema instead of auto-detection.
            By default None, which means the schema will be auto-detected from
            the engine's output names.
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
        no_warn : bool, optional
            If True, suppresses warnings from TensorRT during engine deserialization.
            Default is None, which means warnings will be shown.
        verbose : bool, optional
            Whether or not to log additional information.
            Only covers the initialization phase.

        Raises
        ------
        ValueError
            If an input or output schema string is invalid.

        """
        self._input_schema_override = input_schema
        self._output_schema_override = output_schema

        # preprocessors need _input_schema, which _configure_model() sets mid-super().__init__
        super().__init__(
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_range=input_range,
            preprocessor=preprocessor,
            resize_method=resize_method,
            mean=mean,
            std=std,
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

        # prepend with 'obb_' to avoid conflicts with ImageModel._nvtx_tags
        self._nvtx_tags.update(
            {
                "obb_init": f"obb_detector::init [{self._tag}]",
                "obb_postprocess": f"obb_detector::postprocess [{self._tag}]",
                "obb_run": f"obb_detector::run [{self._tag}]",
                "obb_get_obb_detections": f"obb_detector::get_obb_detections [{self._tag}]",
                "obb_end2end": f"obb_detector::end2end [{self._tag}]",
                "obb__end2end": f"obb_detector::_end2end [{self._tag}]",
                "obb__end2end_graph": f"obb_detector::_end2end_graph [{self._tag}]",
            }
        )

        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["obb_init"])

        self._conf_thres: float = conf_thres
        self._nms_iou: float = nms_iou_thres

        if self._verbose:
            LOG.debug(f"{self._tag}: Input schema: {self._input_schema}")
            LOG.debug(f"{self._tag}: Output schema: {self._output_schema}")

        if self._output_schema == OBBOutputSchema.YOLO:
            self._postprocess_fn = partial(postprocess_yolo_obb, nms_iou_thres=nms_iou_thres)
        else:
            err_msg = f"Unsupported OBB output schema: {self._output_schema}"
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()  # init
            raise ValueError(err_msg)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # init

    @property
    def input_schema(self: Self) -> InputSchema:
        """Get the input schema used by this model."""
        return self._input_schema

    @property
    def output_schema(self: Self) -> OBBOutputSchema:
        """Get the output schema used by this model."""
        return self._output_schema

    def _configure_model(self: Self) -> None:
        """Auto-detect or apply input/output schemas from the loaded engine."""
        self._input_schema, self._output_schema = resolve_obb_schemas(
            self._engine,
            self._input_schema_override,
            self._output_schema_override,
        )

    def postprocess(
        self: Self,
        outputs: list[np.ndarray],
        ratios: list[tuple[float, float]],
        padding: list[tuple[float, float]],
        conf_thres: float | None = None,
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[list[np.ndarray]]:
        """
        Postprocess the outputs.

        Parameters
        ----------
        outputs : list[np.ndarray]
            The raw outputs from the engine to postprocess.
        ratios : list[tuple[float, float]]
            The rescale ratios used during preprocessing for each image.
        padding : list[tuple[float, float]]
            The padding values used during preprocessing for each image.
        conf_thres : float, optional
            The confidence threshold to filter detections by.
            If not passed, will use value from constructor.
        no_copy : bool, optional
            If True, do not copy the data from the allocated
            memory. If the data is not copied, it WILL BE
            OVERWRITTEN INPLACE once new data is generated.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[list[np.ndarray]]
            The postprocessed outputs per image, each containing
            [bboxes, scores, class_ids, rboxes].

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["obb_postprocess"])

        if verbose:
            LOG.debug(f"{self._tag}: postprocess")

        conf_thres = conf_thres if conf_thres is not None else self._conf_thres
        t0 = time.perf_counter()
        data = self._postprocess_fn(
            outputs,
            ratios=ratios,
            padding=padding,
            conf_thres=conf_thres,
            input_size=self._input_size,
            no_copy=no_copy,
            verbose=verbose,
        )
        t1 = time.perf_counter()
        self._post_profile = (t0, t1)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # postprocess

        return data

    @overload
    def __call__(
        self: Self,
        images: np.ndarray,
        ratios: tuple[float, float] | None = ...,
        padding: tuple[float, float] | None = ...,
        conf_thres: float | None = ...,
        *,
        preprocessed: bool | None = ...,
        postprocess: bool | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    @overload
    def __call__(
        self: Self,
        images: list[np.ndarray],
        ratios: list[tuple[float, float]] | None = ...,
        padding: list[tuple[float, float]] | None = ...,
        conf_thres: float | None = ...,
        *,
        preprocessed: bool | None = ...,
        postprocess: bool | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray] | list[list[np.ndarray]]: ...

    def __call__(
        self: Self,
        images: np.ndarray | list[np.ndarray],
        ratios: tuple[float, float] | list[tuple[float, float]] | None = None,
        padding: tuple[float, float] | list[tuple[float, float]] | None = None,
        conf_thres: float | None = None,
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[np.ndarray] | list[list[np.ndarray]]:
        """
        Run the model on input.

        Parameters
        ----------
        images : np.ndarray | list[np.ndarray]
            A single image (HWC format) or list of images to run the model on.
        ratios : tuple[float, float] | list[tuple[float, float]], optional
            The ratios generated during preprocessing. For single image, pass tuple.
            For batch, pass list.
        padding : tuple[float, float] | list[tuple[float, float]], optional
            The padding values used during preprocessing. For single image, pass tuple.
            For batch, pass list.
        conf_thres : float, optional
            Optional confidence threshold to filter detections
            via during postprocessing.
        preprocessed : bool, optional
            Whether or not the inputs have been preprocessed.
            If None, will preprocess inputs.
        postprocess : bool, optional
            Whether or not to postprocess the outputs.
            If None, will postprocess outputs.
        no_copy : bool, optional
            If True, the outputs will not be copied out
            from the cuda allocated host memory. Instead,
            the host memory will be returned directly.
            This memory WILL BE OVERWRITTEN INPLACE by
            future inferences.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[np.ndarray] | list[list[np.ndarray]]
            The outputs. For single image with postprocess=True,
            returns list[np.ndarray]. For batch, returns batch results.

        """
        return self.run(  # ty: ignore[no-matching-overload]
            images,
            ratios,
            padding,
            conf_thres,
            preprocessed=preprocessed,
            postprocess=postprocess,
            no_copy=no_copy,
            verbose=verbose,
        )

    @overload
    def run(
        self: Self,
        images: list[np.ndarray],
        ratios: list[tuple[float, float]] | None = ...,
        padding: list[tuple[float, float]] | None = ...,
        conf_thres: float | None = ...,
        *,
        preprocessed: bool | None = ...,
        postprocess: Literal[False],
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    @overload
    def run(
        self: Self,
        images: list[np.ndarray],
        ratios: list[tuple[float, float]] | None = ...,
        padding: list[tuple[float, float]] | None = ...,
        conf_thres: float | None = ...,
        *,
        preprocessed: bool | None = ...,
        postprocess: Literal[True] | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[list[np.ndarray]]: ...

    @overload
    def run(
        self: Self,
        images: list[np.ndarray],
        ratios: list[tuple[float, float]] | None = ...,
        padding: list[tuple[float, float]] | None = ...,
        conf_thres: float | None = ...,
        *,
        preprocessed: bool | None = ...,
        postprocess: bool | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray] | list[list[np.ndarray]]: ...

    @overload
    def run(
        self: Self,
        images: np.ndarray,
        ratios: tuple[float, float] | None = ...,
        padding: tuple[float, float] | None = ...,
        conf_thres: float | None = ...,
        *,
        preprocessed: bool | None = ...,
        postprocess: Literal[False],
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    @overload
    def run(
        self: Self,
        images: np.ndarray,
        ratios: tuple[float, float] | None = ...,
        padding: tuple[float, float] | None = ...,
        conf_thres: float | None = ...,
        *,
        preprocessed: bool | None = ...,
        postprocess: Literal[True] | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    @overload
    def run(
        self: Self,
        images: np.ndarray,
        ratios: tuple[float, float] | None = ...,
        padding: tuple[float, float] | None = ...,
        conf_thres: float | None = ...,
        *,
        preprocessed: bool | None = ...,
        postprocess: bool | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    def run(
        self: Self,
        images: np.ndarray | list[np.ndarray],
        ratios: tuple[float, float] | list[tuple[float, float]] | None = None,
        padding: tuple[float, float] | list[tuple[float, float]] | None = None,
        conf_thres: float | None = None,
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[np.ndarray] | list[list[np.ndarray]]:
        """
        Run the model on input.

        Parameters
        ----------
        images : np.ndarray | list[np.ndarray]
            A single image (HWC format) or list of images to run the model on.
        ratios : tuple[float, float] | list[tuple[float, float]], optional
            The ratios generated during preprocessing. For single image, pass tuple.
            For batch, pass list.
        padding : tuple[float, float] | list[tuple[float, float]], optional
            The padding values used during preprocessing. For single image, pass tuple.
            For batch, pass list.
        conf_thres : float, optional
            Optional confidence threshold to filter detections
            via during postprocessing.
        preprocessed : bool, optional
            Whether or not the inputs have been preprocessed.
            If None, will preprocess inputs.
        postprocess : bool, optional
            Whether or not to postprocess the outputs.
            If None, will postprocess outputs.
            If postprocessing will occur and the inputs were
            passed already preprocessed, then the ratios and
            padding must be passed for postprocessing.
        no_copy : bool, optional
            If True, the outputs will not be copied out
            from the cuda allocated host memory. Instead,
            the host memory will be returned directly.
            This memory WILL BE OVERWRITTEN INPLACE by
            future inferences.
            In special case where, preprocessing and
            postprocessing will occur during run and no_copy
            was not passed (is None), then no_copy will be used
            for preprocessing and inference stages.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[np.ndarray] | list[list[np.ndarray]]
            For single image with postprocess=True: list[np.ndarray] (single image outputs).
            For batch with postprocess=True: list[list[np.ndarray]] (per-image outputs).
            For postprocess=False: list[np.ndarray] (raw outputs).

        Raises
        ------
        RuntimeError
            If postprocessing is running, but ratios/padding not found
        ValueError
            If preprocessed inputs are not a single batch tensor.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["obb_run"])

        if verbose:
            LOG.debug(f"{self._tag}: run")

        if isinstance(images, np.ndarray):
            batch_images: list[np.ndarray] = [images]
            is_single = True
        else:
            batch_images = images
            is_single = False

        batch_ratios: list[tuple[float, float]] | None
        batch_padding: list[tuple[float, float]] | None
        if ratios is not None and isinstance(ratios, tuple) and isinstance(ratios[0], float):
            batch_ratios = [ratios]
        elif isinstance(ratios, list):
            batch_ratios = ratios
        else:
            batch_ratios = ratios  # ty: ignore[invalid-assignment]
        if (
            padding is not None
            and isinstance(padding, tuple)
            and isinstance(padding[0], (int, float))
        ):
            batch_padding = [padding]
        elif isinstance(padding, list):
            batch_padding = padding
        else:
            batch_padding = padding

        if preprocessed is None:
            preprocessed = False
        if postprocess is None:
            postprocess = True

        if no_copy is None and not preprocessed and postprocess:
            # elide two copies when preprocess/run/postprocess happen in one call
            no_copy_pre: bool | None = True
            no_copy_run: bool | None = True
            no_copy_post: bool | None = False
        else:
            no_copy_pre = no_copy
            no_copy_run = no_copy
            no_copy_post = no_copy

        if verbose:
            LOG.debug(
                f"{self._tag}: Running: preprocessed: {preprocessed}, postprocess: {postprocess}",
            )

        if not preprocessed:
            if verbose:
                LOG.debug("Preprocessing inputs")
            tensor, batch_ratios, batch_padding = self.preprocess(batch_images, no_copy=no_copy_pre)
        else:
            if len(batch_images) != 1:
                err_msg = "Preprocessed inputs must be a list containing a single batch tensor."
                if FLAGS.NVTX_ENABLED:
                    nvtx.pop_range()  # run
                raise ValueError(err_msg)
            tensor = batch_images[0]

        batch_size = len(batch_images) if not preprocessed else tensor.shape[0]

        engine_inputs = [tensor]
        if self._use_image_size:
            orig_sizes = np.array(
                [img.shape[:2] for img in batch_images]
                if not preprocessed
                else [(self._input_size[1], self._input_size[0])] * batch_size,
                dtype=np.int32,
            )
            engine_inputs.append(orig_sizes)
        if self._use_scale_factor:
            scale_factors = np.array(batch_ratios, dtype=np.float32)
            engine_inputs.append(scale_factors)

        t0 = time.perf_counter()
        outputs: list[np.ndarray] = self._engine(engine_inputs, no_copy=no_copy_run)
        t1 = time.perf_counter()

        if postprocess:
            if verbose:
                LOG.debug("Postprocessing outputs")
            if batch_ratios is None or batch_padding is None:
                err_msg = "Must pass ratios/padding if postprocessing and passing already preprocessed inputs."
                if FLAGS.NVTX_ENABLED:
                    nvtx.pop_range()  # run
                raise RuntimeError(err_msg)
            postprocessed_outputs = self.postprocess(
                outputs,
                batch_ratios,
                batch_padding,
                conf_thres,
                no_copy=no_copy_post,
            )
            self._infer_profile = (t0, t1)

            if is_single:
                if FLAGS.NVTX_ENABLED:
                    nvtx.pop_range()  # run
                return postprocessed_outputs[0]
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()  # run
            return postprocessed_outputs

        self._infer_profile = (t0, t1)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # run

        return outputs

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

    def end2end(
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

    def _end2end(
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

    def _end2end_graph(
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
