# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, overload

import numpy as np
import nvtx
from typing_extensions import Literal

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

if TYPE_CHECKING:
    from pathlib import Path

    from typing_extensions import Self

    from .interfaces import ImageInput


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
            When enabled, CUDA graphs are used both at the engine level and for
            end-to-end execution in the end2end() method. The first call to
            end2end() will capture a CUDA graph of the full preprocessing +
            inference pipeline, and subsequent calls will replay it. Graphs
            are cached per batch size and input buffer set, so image
            resolution and batch size may both change between calls.
            Only effective with async_v3 backend. Default is True.
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
            device=device,
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
                "det_get_detections": f"detector::get_detections [{self._tag}]",
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
        # RT_DETR_V3 and schemas using image-size/scale-factor build extra
        # inputs from host data in _engine_inputs, so those keep the host
        # path on run(); everything else can take the direct-GPU path.
        self._single_input_schema = (
            self._input_schema != InputSchema.RT_DETR_V3
            and not self._use_image_size
            and not self._use_scale_factor
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
        images: ImageInput,
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
        images: list[ImageInput],
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
        images: ImageInput | list[ImageInput],
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
        images : ImageInput | list[ImageInput]
            A single image or list of images to run the model on, each an
            HWC uint8 ``np.ndarray`` or a ``Buffer`` (host or device)
            holding one.
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

    # run overloads - batch input (3 overloads)
    @overload
    def run(
        self: Self,
        images: list[ImageInput],
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
        images: list[ImageInput],
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
        images: list[ImageInput],
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
        images: ImageInput,
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
        images: ImageInput,
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
        images: ImageInput,
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
        images: ImageInput | list[ImageInput],
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
        images : ImageInput | list[ImageInput]
            A single image or list of images to run the model on, each an
            HWC uint8 ``np.ndarray`` or a ``Buffer`` (host or device)
            holding one.
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
        return self._run_core(
            images,
            ratios,
            padding,
            preprocessed=preprocessed,
            postprocess=postprocess,
            no_copy=no_copy,
            verbose=verbose,
            post=lambda o, r, p, nc: self.postprocess(
                o, r, p, conf_thres, no_copy=nc, verbose=verbose
            ),
        )

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

        conf_thres = conf_thres or self._conf_thres
        nms_iou = nms_iou_thres or self._nms_iou
        use_nms = extra_nms if extra_nms is not None else self._nms
        agnostic = agnostic_nms if agnostic_nms is not None else self._agnostic_nms

        batch, single = self._as_batch(outputs)
        result = get_detections(
            batch,
            conf_thres=conf_thres,
            nms_iou_thres=nms_iou,
            extra_nms=use_nms,
            agnostic_nms=agnostic,
            verbose=verbose,
        )

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # get_detections

        return result[0] if single else result

    # end2end overloads
    @overload
    def end2end(
        self: Self,
        images: ImageInput,
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
        images: list[ImageInput],
        conf_thres: float | None = ...,
        nms_iou_thres: float | None = ...,
        *,
        extra_nms: bool | None = ...,
        agnostic_nms: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[list[tuple[tuple[int, int, int, int], float, int]]]: ...

    def end2end(
        self: Self,
        images: ImageInput | list[ImageInput],
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
        images : ImageInput | list[ImageInput]
            A single image or list of images to perform inference with, each
            an HWC uint8 ``np.ndarray`` or a ``Buffer`` (host or device)
            holding one.
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
            If end2end_graph is enabled and a static engine receives a batch size different than what it was built for.
        RuntimeError
            If end2end_graph is enabled and CUDA graph capture fails.

        """
        return self._end2end_core(
            images,
            verbose=verbose,
            post=lambda o, r, p, nc: self.postprocess(
                o, r, p, conf_thres, no_copy=nc, verbose=verbose
            ),
            get=lambda pp: self.get_detections(
                pp,
                conf_thres=conf_thres,
                nms_iou_thres=nms_iou_thres,
                extra_nms=extra_nms,
                agnostic_nms=agnostic_nms,
                verbose=verbose,
            ),
        )

    def _engine_inputs(
        self: Self,
        tensor: np.ndarray,
        images: list[ImageInput],
        ratios: list[tuple[float, float]] | None,
        *,
        preprocessed: bool,
    ) -> list[np.ndarray]:
        """Build host engine inputs, adding orig size / scale factor arrays per input schema."""
        if preprocessed:
            sizes = [(self._input_size[1], self._input_size[0])] * tensor.shape[0]
        else:
            sizes = [img.shape[:2] for img in images]
        extras: list[np.ndarray] = []
        if self._use_image_size:
            extras.append(np.array(sizes, dtype=self._orig_size_dtype))
        if self._use_scale_factor:
            extras.append(np.array(ratios, dtype=np.float32))
        return self._build_graph_input_ptrs(tensor, extras)

    def _prepare_extra_engine_inputs_gpu(self: Self) -> list[int]:
        """Return additional GPU input pointers for DETR-style models (GPU preprocessor path)."""
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["det__prepare_extra_gpu"])

        input_ptrs: list[int] = []
        if self._use_image_size:
            orig_size_ptr, valid = self._preprocessor.orig_size_allocation  # ty: ignore[unresolved-attribute]
            if not valid:
                err_msg = "orig_image_size buffer not valid"
                if FLAGS.NVTX_ENABLED:
                    nvtx.pop_range()  # prepare_extra_gpu
                raise RuntimeError(err_msg)
            input_ptrs.append(orig_size_ptr)
        if self._use_scale_factor:
            scale_ptr, scale_valid = self._preprocessor.scale_factor_allocation  # ty: ignore[unresolved-attribute]
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
        images: list[ImageInput],
        ratios: list[tuple[float, float]],
    ) -> list[int]:
        """Return additional GPU input pointers for DETR-style models (CPU preprocessor path)."""
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["det__prepare_extra_cpu"])

        input_ptrs: list[int] = []
        input_idx = 1  # Start after the main image input

        if self._use_image_size:
            # Build orig_target_sizes: (batch, 2) with (height, width) per image
            orig_sizes = np.array(
                [img.shape[:2] for img in images],
                dtype=self._orig_size_dtype,
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
        image: Any,  # noqa: ANN401
        extras: list[Any],
    ) -> list[Any]:
        """Override to handle RTDETRv3 input ordering (im_shape, image, scale_factor)."""
        # RTDETRv3 expects: (im_shape, image, scale_factor)
        # extras[0] = orig_size, extras[1] = scale_factor
        if self._input_schema == InputSchema.RT_DETR_V3 and len(extras) >= 2:  # noqa: PLR2004
            return [extras[0], image, extras[1]]
        # Default: image first, then extra inputs
        return [image, *extras]
