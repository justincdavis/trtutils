# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import time
from typing import TYPE_CHECKING, overload

import numpy as np
import nvtx
from typing_extensions import Literal, TypeGuard

from trtutils._flags import FLAGS
from trtutils._log import LOG
from trtutils.core._memory import memcpy_host_to_device_async

from ._image_model import ImageModel
from ._schema import InputSchema, OutputSchema, resolve_detector_schemas
from .interfaces import DetectorInterface
from .postprocessors import (
    get_detections,
    postprocess_detr,
    postprocess_detr_lbs,
    postprocess_efficient_nms,
    postprocess_rfdetr,
    postprocess_rtdetrv3,
    postprocess_yolov10,
)
from .preprocessors import CUDAPreprocessor, TRTPreprocessor
from .preprocessors._image_preproc import _is_single_image

if TYPE_CHECKING:
    from pathlib import Path

    from typing_extensions import Self


def _is_postprocessed_outputs(
    outputs: list[np.ndarray] | list[list[np.ndarray]],
) -> TypeGuard[list[list[np.ndarray]]]:
    return not outputs or isinstance(outputs[0], list)


_TUPLE_PAIR_LEN = 2  # (ratio_x, ratio_y) or (pad_x, pad_y) tuples


class Detector(ImageModel, DetectorInterface):
    """Implementation of object detectors."""

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
        output_schema: OutputSchema | str | None = None,
        dla_core: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = False,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        """
        Create a Detector object.

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
        conf_thres : float, optional
            The confidence threshold above which to generate detections.
            By default 0.1
        nms_iou_thres : float, optional
            The IOU threshold to use the in the optional and additnal
            NMS operation. By default, 0.5
        mean : tuple[float, float, float] | None, optional
            The mean values to use for the imagenet normalization.
            By default, None, which means no normalization will be applied.
        std : tuple[float, float, float] | None, optional
            The standard deviation values to use for the imagenet normalization.
            By default, None, which means no normalization will be applied.
        input_schema : InputSchema, str, optional
            Manually specify the input schema instead of auto-detection.
            Can be an InputSchema enum value or a string matching the enum name
            (e.g., "YOLO", "RT_DETR", "RT_DETR_V3", "RF_DETR").
            By default None, which means the schema will be auto-detected from
            the engine's input names.
        output_schema : OutputSchema, str, optional
            Manually specify the output schema instead of auto-detection.
            Can be an OutputSchema enum value or a string matching the enum name
            (e.g., "EFFICIENT_NMS", "YOLO_V10", "DETR", "RF_DETR").
            By default None, which means the schema will be auto-detected from
            the engine's output names.
        dla_core : int, optional
            The DLA core to assign DLA layers of the engine to. Default is None.
            If None, any DLA layers will be assigned to DLA core 0.
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
            When enabled, CUDA graphs are used both at the engine level and for
            end-to-end execution in the end2end() method. The first call to
            end2end() will capture a CUDA graph of the full preprocessing +
            inference pipeline, and subsequent calls will replay it. Input
            dimensions are locked after the first end2end() call.
            Only effective with async_v3 backend. Default is None (disabled).
        extra_nms : bool, optional
            Whether or not an additional CPU-side NMS operation
            should be conducted on final detections.
        agnostic_nms : bool, optional
            Whether or not the optional/additional NMS operation
            should perform class agnostic NMS.
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
        # store user-provided schema overrides for _configure_model to use
        self._input_schema_override = input_schema
        self._output_schema_override = output_schema

        # parent creates engine, calls _configure_model() (which sets schemas),
        # then creates preprocessors (which need _input_schema for dtype)
        super().__init__(
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_range=input_range,
            preprocessor=preprocessor,
            resize_method=resize_method,
            mean=mean,
            std=std,
            dla_core=dla_core,
            backend=backend,
            warmup=warmup,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
            cuda_graph=cuda_graph,
            no_warn=no_warn,
            verbose=verbose,
        )

        # prepend with 'det_' to avoid conflicts with ImageModel._nvtx_tags
        self._nvtx_tags.update(
            {
                "det_init": f"detector::init [{self._tag}]",
                "det_postprocess": f"detector::postprocess [{self._tag}]",
                "det_run": f"detector::run [{self._tag}]",
                "det_get_detections": f"detector::get_detections [{self._tag}]",
                "det_end2end": f"detector::end2end [{self._tag}]",
                "det__end2end": f"detector::_end2end [{self._tag}]",
                "det__end2end_graph": f"detector::_end2end_graph [{self._tag}]",
                "det__prepare_extra_gpu": f"detector::_prepare_extra_gpu [{self._tag}]",
                "det__prepare_extra_cpu": f"detector::_prepare_extra_cpu [{self._tag}]",
            }
        )

        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["det_init"])

        self._conf_thres: float = conf_thres
        self._nms_iou: float = nms_iou_thres
        self._nms: bool | None = extra_nms
        self._agnostic_nms: bool | None = agnostic_nms

        if self._verbose:
            LOG.debug(f"{self._tag}: Input schema: {self._input_schema}")
            LOG.debug(f"{self._tag}: Output schema: {self._output_schema}")

        # solve for the postprocessing function
        if self._output_schema == OutputSchema.YOLO_V10:
            self._postprocess_fn = postprocess_yolov10
        elif self._output_schema == OutputSchema.RF_DETR:
            self._postprocess_fn = postprocess_rfdetr
        elif self._output_schema == OutputSchema.DETR:
            self._postprocess_fn = postprocess_detr
        elif self._output_schema == OutputSchema.DETR_LBS:
            self._postprocess_fn = postprocess_detr_lbs
        elif self._output_schema == OutputSchema.RT_DETR_V3:
            self._postprocess_fn = postprocess_rtdetrv3
        else:
            self._postprocess_fn = postprocess_efficient_nms

        # use unified get detections function
        self._get_detections_fn = get_detections

        if self._verbose:
            LOG.debug(f"{self._tag}: Using image size: {self._use_image_size}")
            LOG.debug(f"{self._tag}: Using scale factor: {self._use_scale_factor}")

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # init

    @property
    def input_schema(self: Self) -> InputSchema:
        """Get the input schema used by this detector."""
        return self._input_schema

    @property
    def output_schema(self: Self) -> OutputSchema:
        """Get the output schema used by this detector."""
        return self._output_schema

    def _configure_model(self: Self) -> None:
        """Auto-detect or apply input/output schemas from the loaded engine."""
        self._input_schema, self._output_schema = resolve_detector_schemas(
            self._engine,
            self._input_schema_override,
            self._output_schema_override,
        )
        self._use_image_size = self._input_schema.uses_image_size
        self._use_scale_factor = self._input_schema.uses_scale_factor
        self._orig_size_dtype = self._input_schema.orig_size_dtype

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
            [bboxes, scores, class_ids].

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["det_postprocess"])

        if verbose:
            LOG.debug(f"{self._tag}: postprocess")

        conf_thres = conf_thres or self._conf_thres
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

    # __call__ overloads
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
        return self.run(  # type: ignore[call-overload]
            images,
            ratios,
            padding,
            conf_thres,
            preprocessed=preprocessed,
            postprocess=postprocess,
            no_copy=no_copy,
            verbose=verbose,
        )

    # run overloads - batch input (3 overloads)
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

    # run overloads - single image input (3 overloads)
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
            nvtx.push_range(self._nvtx_tags["det_run"])

        if verbose:
            LOG.debug(f"{self._tag}: run")

        # Handle single-image input
        is_single = _is_single_image(images)
        if is_single:
            images = [images]  # type: ignore[list-item]
            # Wrap single ratios/padding to lists
            if (
                ratios is not None
                and isinstance(ratios, tuple)
                and len(ratios) == _TUPLE_PAIR_LEN
                and isinstance(ratios[0], float)
            ):
                ratios = [ratios]
            if (
                padding is not None
                and isinstance(padding, tuple)
                and len(padding) == _TUPLE_PAIR_LEN
                and isinstance(padding[0], (int, float))
            ):
                padding = [padding]

        # assign flags
        if preprocessed is None:
            preprocessed = False
        if postprocess is None:
            postprocess = True

        # assign no_copy values
        if no_copy is None and not preprocessed and postprocess:
            # remove two sets of copies when doing preprocess/run/postprocess inside
            # a single run call
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

        # handle preprocessing
        if not preprocessed:
            if verbose:
                LOG.debug("Preprocessing inputs")
            tensor, ratios, padding = self.preprocess(images, no_copy=no_copy_pre)
        else:
            # images is already preprocessed tensor when preprocessed=True
            if len(images) != 1:
                err_msg = "Preprocessed inputs must be a list containing a single batch tensor."
                if FLAGS.NVTX_ENABLED:
                    nvtx.pop_range()  # run
                raise ValueError(err_msg)
            tensor = images[0]

        batch_size = len(images) if not preprocessed else tensor.shape[0]

        # build input list based on schema
        # RT_DETR_V3 expects (im_shape, image, scale_factor) order
        # RT_DETR expects (images, orig_target_sizes) order
        if self._input_schema == InputSchema.RT_DETR_V3:
            # Build im_shape: (batch, 2) with (height, width) per image
            im_shape = np.array(
                [img.shape[:2] for img in images]
                if not preprocessed
                else [(self._input_size[1], self._input_size[0])] * batch_size,
                dtype=np.float32,
            )
            # Build scale_factor: (batch, 2) from ratios list
            scale_factors = np.array(ratios, dtype=np.float32)
            engine_inputs = [im_shape, tensor, scale_factors]
        else:
            engine_inputs = [tensor]
            if self._use_image_size:
                # Build batched orig_target_sizes: (batch, 2) with (height, width) per image
                orig_sizes = np.array(
                    [img.shape[:2] for img in images]
                    if not preprocessed
                    else [(self._input_size[1], self._input_size[0])] * batch_size,
                    dtype=np.int32,
                )
                engine_inputs.append(orig_sizes)
            if self._use_scale_factor:
                # Build batched scale_factor: (batch, 2) from ratios list
                scale_factors = np.array(ratios, dtype=np.float32)
                engine_inputs.append(scale_factors)

        # execute
        t0 = time.perf_counter()
        outputs: list[np.ndarray] = self._engine(engine_inputs, no_copy=no_copy_run)
        t1 = time.perf_counter()

        # handle postprocessing
        if postprocess:
            if verbose:
                LOG.debug("Postprocessing outputs")
            if ratios is None or padding is None:
                err_msg = "Must pass ratios/padding if postprocessing and passing already preprocessed inputs."
                if FLAGS.NVTX_ENABLED:
                    nvtx.pop_range()  # run
                raise RuntimeError(err_msg)
            postprocessed_outputs = self.postprocess(
                outputs,
                ratios,  # type: ignore[arg-type]
                padding,  # type: ignore[arg-type]
                conf_thres,
                no_copy=no_copy_post,
            )
            self._infer_profile = (t0, t1)

            # Unwrap for single-image input
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

    # get_detections overloads
    @overload
    def get_detections(
        self: Self,
        outputs: list[np.ndarray],
        conf_thres: float | None = ...,
        nms_iou_thres: float | None = ...,
        *,
        extra_nms: bool | None = ...,
        agnostic_nms: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[tuple[tuple[int, int, int, int], float, int]]: ...

    @overload
    def get_detections(
        self: Self,
        outputs: list[list[np.ndarray]],
        conf_thres: float | None = ...,
        nms_iou_thres: float | None = ...,
        *,
        extra_nms: bool | None = ...,
        agnostic_nms: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[list[tuple[tuple[int, int, int, int], float, int]]]: ...

    def get_detections(
        self: Self,
        outputs: list[np.ndarray] | list[list[np.ndarray]],
        conf_thres: float | None = None,
        nms_iou_thres: float | None = None,
        *,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
        verbose: bool | None = None,
    ) -> (
        list[tuple[tuple[int, int, int, int], float, int]]
        | list[list[tuple[tuple[int, int, int, int], float, int]]]
    ):
        """
        Get the bounding boxes from postprocessed outputs.

        Parameters
        ----------
        outputs : list[np.ndarray] | list[list[np.ndarray]]
            For single image: list[np.ndarray] (single image's postprocessed outputs).
            For batch: list[list[np.ndarray]] (postprocessed outputs per image).
        conf_thres : float, optional
            The confidence threshold with which to retrieve bounding boxes.
            By default None, which will use value passed during initialization.
        nms_iou_thres : float
            The IOU threshold to use during the optional/additional
            NMS operation. By default, None which will use value
            provided during initialization.
        extra_nms : bool, optional
            Whether or not to perform an additional NMS operation.
            By default None, which will use value provided during
            initialization.
        agnostic_nms: bool, optional
            Whether or not to perform class-agnostic NMS for the
            optional/additional operation. By default None, which
            will use value provided during initialization.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[tuple[...]] | list[list[tuple[...]]]
            For single image: list[tuple[...]] (detections for single image).
            For batch: list[list[tuple[...]]] (detections per image).

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["det_get_detections"])

        if verbose:
            LOG.debug(f"{self._tag}: get_detections")

        # Detect if this is single-image output (list[np.ndarray]) vs batch (list[list[np.ndarray]])
        is_single = outputs and isinstance(outputs[0], np.ndarray)

        conf_thres = conf_thres or self._conf_thres
        nms_iou = nms_iou_thres or self._nms_iou
        use_nms = extra_nms if extra_nms is not None else self._nms
        agnostic = agnostic_nms if agnostic_nms is not None else self._agnostic_nms

        if is_single:
            # Wrap single image outputs for batch processing
            batch_outputs: list[list[np.ndarray]] = [outputs]  # type: ignore[list-item]
            result = self._get_detections_fn(
                batch_outputs,
                conf_thres=conf_thres,
                nms_iou_thres=nms_iou,
                extra_nms=use_nms,
                agnostic_nms=agnostic,
                verbose=verbose,
            )
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()  # get_detections
            return result[0]  # Unwrap

        result_batch = self._get_detections_fn(
            outputs,  # type: ignore[arg-type]
            conf_thres=conf_thres,
            nms_iou_thres=nms_iou,
            extra_nms=use_nms,
            agnostic_nms=agnostic,
            verbose=verbose,
        )

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # get_detections

        return result_batch

    # end2end overloads
    @overload
    def end2end(
        self: Self,
        images: np.ndarray,
        conf_thres: float | None = ...,
        nms_iou_thres: float | None = ...,
        *,
        extra_nms: bool | None = ...,
        agnostic_nms: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[tuple[tuple[int, int, int, int], float, int]]: ...

    @overload
    def end2end(
        self: Self,
        images: list[np.ndarray],
        conf_thres: float | None = ...,
        nms_iou_thres: float | None = ...,
        *,
        extra_nms: bool | None = ...,
        agnostic_nms: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[list[tuple[tuple[int, int, int, int], float, int]]]: ...

    def end2end(
        self: Self,
        images: np.ndarray | list[np.ndarray],
        conf_thres: float | None = None,
        nms_iou_thres: float | None = None,
        *,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
        verbose: bool | None = None,
    ) -> (
        list[tuple[tuple[int, int, int, int], float, int]]
        | list[list[tuple[tuple[int, int, int, int], float, int]]]
    ):
        """
        Perform end to end inference for a batch of images.

        Equivalent to running preprocess, run, postprocess, and
        get_detections in that order. Makes some memory transfer
        optimizations under the hood to improve performance.

        Parameters
        ----------
        images : np.ndarray | list[np.ndarray]
            A single image (HWC format) or list of images to perform inference with.
        conf_thres : float, optional
            The confidence threshold with which to retrieve bounding boxes.
            By default None.
        nms_iou_thres : float
            The IOU threshold to use during the optional/additional
            NMS operation. By default, None which will use value
            provided during initialization.
        extra_nms : bool, optional
            Whether or not to perform an additional NMS operation.
            By default None, which will use value provided during
            initialization.
        agnostic_nms: bool, optional
            Whether or not to perform class-agnostic NMS for the
            optional/additional operation. By default None, which
            will use value provided during initialization.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[tuple[...]] | list[list[tuple[...]]]
            For single image: list[tuple[...]] (detections).
            For batch: list[list[tuple[...]]] (detections per image).

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
            nvtx.push_range(self._nvtx_tags["det_end2end"])

        if verbose:
            LOG.debug(f"{self._tag}: end2end")

        # Handle single-image input
        is_single = _is_single_image(images)
        if is_single:
            images = [images]  # type: ignore[list-item]

        # Dispatch based on graph flag
        if self._e2e_graph_enabled:
            result = self._end2end_graph(
                images,  # type: ignore[arg-type]
                conf_thres=conf_thres,
                nms_iou_thres=nms_iou_thres,
                extra_nms=extra_nms,
                agnostic_nms=agnostic_nms,
                verbose=verbose,
            )
        else:
            result = self._end2end(
                images,  # type: ignore[arg-type]
                conf_thres=conf_thres,
                nms_iou_thres=nms_iou_thres,
                extra_nms=extra_nms,
                agnostic_nms=agnostic_nms,
                verbose=verbose,
            )

        # Unwrap for single-image input
        if is_single:
            # result is already unwrapped from get_detections for single image
            # but we need to check if it was processed as batch
            if isinstance(result, list) and result and isinstance(result[0], list):
                if FLAGS.NVTX_ENABLED:
                    nvtx.pop_range()  # end2end
                return result[0]
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()  # end2end
            return result

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # end2end

        return result

    def _end2end(
        self: Self,
        images: list[np.ndarray],
        conf_thres: float | None = None,
        nms_iou_thres: float | None = None,
        *,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
        verbose: bool | None = None,
    ) -> list[list[tuple[tuple[int, int, int, int], float, int]]]:
        """Execute the standard end2end path without graph capture."""
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["det__end2end"])

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
                err_msg = "Expected postprocessed detector outputs in end2end."
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

            # Build input pointers based on InputSchema
            # RT_DETR_V3 expects (im_shape, image, scale_factor) order
            if self._input_schema == InputSchema.RT_DETR_V3:
                orig_size_ptr, valid = self._preprocessor.orig_size_allocation
                if not valid:
                    err_msg = "orig_image_size buffer not valid"
                    if FLAGS.NVTX_ENABLED:
                        nvtx.pop_range()  # _end2end
                    raise RuntimeError(err_msg)
                scale_ptr, scale_valid = self._preprocessor.scale_factor_allocation
                if not scale_valid:
                    err_msg = "scale_factor buffer not valid"
                    if FLAGS.NVTX_ENABLED:
                        nvtx.pop_range()  # _end2end
                    raise RuntimeError(err_msg)
                input_ptrs = [orig_size_ptr, gpu_ptr, scale_ptr]
            else:
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

        # generate the detections
        result = self.get_detections(
            postprocessed,
            conf_thres=conf_thres,
            nms_iou_thres=nms_iou_thres,
            extra_nms=extra_nms,
            agnostic_nms=agnostic_nms,
            verbose=verbose,
        )

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # _end2end

        return result

    def _prepare_extra_engine_inputs_gpu(self: Self) -> list[int]:
        """Return additional GPU input pointers for DETR-style models (GPU preprocessor path)."""
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["det__prepare_extra_gpu"])

        input_ptrs: list[int] = []
        if self._use_image_size:
            orig_size_ptr, valid = self._preprocessor.orig_size_allocation  # type: ignore[union-attr]
            if not valid:
                err_msg = "orig_image_size buffer not valid"
                if FLAGS.NVTX_ENABLED:
                    nvtx.pop_range()  # prepare_extra_gpu
                raise RuntimeError(err_msg)
            input_ptrs.append(orig_size_ptr)
        if self._use_scale_factor:
            scale_ptr, scale_valid = self._preprocessor.scale_factor_allocation  # type: ignore[union-attr]
            if not scale_valid:
                err_msg = "scale_factor buffer not valid"
                if FLAGS.NVTX_ENABLED:
                    nvtx.pop_range()  # prepare_extra_gpu
                raise RuntimeError(err_msg)
            input_ptrs.append(scale_ptr)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # prepare_extra_gpu

        return input_ptrs

    def _prepare_extra_engine_inputs_cpu(
        self: Self,
        images: list[np.ndarray],
        ratios: list[tuple[float, float]],
    ) -> list[int]:
        """Return additional GPU input pointers for DETR-style models (CPU preprocessor path)."""
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["det__prepare_extra_cpu"])

        input_ptrs: list[int] = []
        input_idx = 1  # Start after the main image input

        if self._use_image_size:
            # Build orig_target_sizes: (batch, 2) with (height, width) per image
            # RTDETRv3 expects float32 for im_shape, other schemas use int32
            orig_size_dtype = (
                np.float32 if self._input_schema == InputSchema.RT_DETR_V3 else np.int32
            )
            orig_sizes = np.array(
                [img.shape[:2] for img in images],
                dtype=orig_size_dtype,
            )
            memcpy_host_to_device_async(
                self._engine._inputs[input_idx].allocation,  # noqa: SLF001
                orig_sizes,
                self._engine.stream,
            )
            input_ptrs.append(self._engine._inputs[input_idx].allocation)  # noqa: SLF001
            input_idx += 1

        if self._use_scale_factor:
            # Build scale_factor from ratios
            scale_factors = np.array(ratios, dtype=np.float32)
            memcpy_host_to_device_async(
                self._engine._inputs[input_idx].allocation,  # noqa: SLF001
                scale_factors,
                self._engine.stream,
            )
            input_ptrs.append(self._engine._inputs[input_idx].allocation)  # noqa: SLF001

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # prepare_extra_cpu

        return input_ptrs

    def _build_graph_input_ptrs(
        self: Self,
        gpu_ptr: int,
        extra_ptrs: list[int],
    ) -> list[int]:
        """Override to handle RTDETRv3 input ordering (im_shape, image, scale_factor)."""
        # RTDETRv3 expects: (im_shape, image, scale_factor)
        # extra_ptrs[0] = orig_size_ptr, extra_ptrs[1] = scale_ptr
        if self._input_schema == InputSchema.RT_DETR_V3 and len(extra_ptrs) >= 2:  # noqa: PLR2004
            return [extra_ptrs[0], gpu_ptr, extra_ptrs[1]]
        # Default: image first, then extra inputs
        return [gpu_ptr, *extra_ptrs]

    def _end2end_graph(
        self: Self,
        images: list[np.ndarray],
        conf_thres: float | None = None,
        nms_iou_thres: float | None = None,
        *,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
        verbose: bool | None = None,
    ) -> list[list[tuple[tuple[int, int, int, int], float, int]]]:
        """
        Execute graph-accelerated end2end path.

        This implementation captures only TRTEngine inference in the CUDA graph.
        Preprocessing runs outside the graph since H2D copies cannot be captured.
        Supports CPU, CUDA, and TRT preprocessors.
        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["det__end2end_graph"])

        # Use shared core graph execution
        raw_outputs, ratios, padding = self._end2end_graph_core(images, verbose=verbose)

        # CPU postprocessing (Detector-specific)
        postprocessed: list[list[np.ndarray]] = self.postprocess(
            raw_outputs,
            ratios,
            padding,
            conf_thres,
            no_copy=True,
            verbose=verbose,
        )
        result = self.get_detections(
            postprocessed,
            conf_thres=conf_thres,
            nms_iou_thres=nms_iou_thres,
            extra_nms=extra_nms,
            agnostic_nms=agnostic_nms,
            verbose=verbose,
        )

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # _end2end_graph

        return result
