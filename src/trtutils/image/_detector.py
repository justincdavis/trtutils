# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import time
from typing import TYPE_CHECKING, cast, overload

import numpy as np
import nvtx
from typing_extensions import Literal

from trtutils._flags import FLAGS
from trtutils._log import LOG
from trtutils.core._buffer import Buffer

from ._image_model import ImageModel
from ._schema import InputSchema, OutputSchema, resolve_schema
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
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from typing_extensions import Self

    from ._image_model import _PostFn
    from .types import Detection


class Detector(ImageModel, DetectorInterface):
    """Implementation of object detectors."""

    _OUTPUT_SCHEMA = OutputSchema

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
        postprocessor: str = "cpu",
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
        class_map: Mapping[int, int] | None = None,
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
            When enabled, the engine records a CUDA graph for each input
            configuration (batch size, resolution, input buffers) once it
            repeats and replays it afterwards, for run() and end2end() alike.
            Preprocessing and postprocessing run outside the graphs.
            Only effective with async_v3 backend. Default is True.
        postprocessor : str, optional
            Where to postprocess, 'cpu' or 'cuda'. 'cuda' decodes the engine outputs
            on the GPU and copies back only the final results. Default is 'cpu'.
        extra_nms : bool, optional
            Whether or not an additional CPU-side NMS operation
            should be conducted on final detections.
        agnostic_nms : bool, optional
            Whether or not the optional/additional NMS operation
            should perform class agnostic NMS.
        class_map : Mapping[int, int], optional
            Translates the model's class IDs to output class IDs after postprocessing.
            IDs not in the map pass through unchanged. By default no translation is done.
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
        # stored for _configure_model, which runs during the parent init
        self._input_schema_override = input_schema
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
            postprocessor=postprocessor,
            no_warn=no_warn,
            verbose=verbose,
            output_schema=output_schema,
        )

        # prepend with 'det_' to avoid conflicts with ImageModel._nvtx_tags
        self._nvtx_tags.update(
            {
                "det_init": f"detector::init [{self._tag}]",
                "det_postprocess": f"detector::postprocess [{self._tag}]",
                "det_get_detections": f"detector::get_detections [{self._tag}]",
            }
        )

        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["det_init"])

        self._conf_thres: float = conf_thres
        self._nms_iou: float = nms_iou_thres
        self._nms: bool | None = extra_nms
        self._agnostic_nms: bool | None = agnostic_nms
        self._class_map: dict[int, int] | None = dict(class_map) if class_map else None

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
        """Resolve the input schema from the loaded engine."""
        self._input_schema = resolve_schema(
            self._engine, InputSchema, self._input_schema_override, inputs=True
        )
        self._use_image_size = self._input_schema.uses_image_size
        self._use_scale_factor = self._input_schema.uses_scale_factor
        # exports bind the size input as int32, int64 or float32, so read it from the engine
        self._orig_size_dtype = None
        if self._use_image_size:
            size_index = self._engine.input_names.index(self._input_names_without_image()[0])
            self._orig_size_dtype = self._engine.input_spec[size_index][1]

    def _input_names_without_image(self: Self) -> list[str]:
        image_name = self._engine.input_names[self._image_input_index]
        return [name for name in self._input_schema.value if name != image_name]

    def postprocess(
        self: Self,
        outputs: list[np.ndarray] | list[Buffer],
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
        outputs : list[np.ndarray] | list[Buffer]
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
        # only the CUDA postprocessor is handed device outputs
        outputs = cast("list[np.ndarray]", outputs)
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["det_postprocess"])

        if verbose:
            LOG.debug(f"{self._tag}: postprocess")

        conf_thres = conf_thres if conf_thres is not None else self._conf_thres
        t0 = time.perf_counter()
        data = self._postprocess_fn(
            self._order_outputs(outputs),
            ratios=ratios,
            padding=padding,
            conf_thres=conf_thres,
            input_size=self._input_size,
            no_copy=no_copy,
            verbose=verbose,
        )
        if self._class_map is not None:
            for result in data:
                ids = result[2]
                result[2] = np.array([self._class_map.get(int(c), int(c)) for c in ids], ids.dtype)
        t1 = time.perf_counter()
        self._post_profile = (t0, t1)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # postprocess

        return data

    def _post_fn(self: Self, conf_thres: float | None, *, verbose: bool | None) -> _PostFn:
        """Bridge ``_run_core`` / ``_end2end_core`` to :meth:`postprocess`."""
        return lambda o, r, p, nc: self.postprocess(o, r, p, conf_thres, no_copy=nc, verbose=verbose)

    # __call__ overloads
    @overload
    def __call__(
        self: Self,
        images: np.ndarray | Buffer,
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
        images: Sequence[np.ndarray | Buffer],
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
        images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
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
        images : np.ndarray | Buffer | Sequence[np.ndarray | Buffer]
            A single image (HWC format) or list of images to run the model on.
        ratios : tuple[float, float] | list[tuple[float, float]], optional
            The ratios generated during preprocessing. For single image, pass tuple.
            For batch, pass list.
        padding : tuple[float, float] | list[tuple[float, float]], optional
            The padding values used during preprocessing. For single image, pass tuple.
            For batch, pass list.
        conf_thres : float, optional
            Optional confidence threshold to filter detections
            via during postprocessing. By default None, which will
            use value passed during initialization.
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
        return self._run_core(
            images,
            ratios,
            padding,
            preprocessed=preprocessed,
            postprocess=postprocess,
            no_copy=no_copy,
            verbose=verbose,
            post=self._post_fn(conf_thres, verbose=verbose),
        )

    # run overloads - batch input (3 overloads)
    @overload
    def run(
        self: Self,
        images: Sequence[np.ndarray | Buffer],
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
        images: Sequence[np.ndarray | Buffer],
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
        images: Sequence[np.ndarray | Buffer],
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
        images: np.ndarray | Buffer,
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
        images: np.ndarray | Buffer,
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
        images: np.ndarray | Buffer,
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
        images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
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
        images : np.ndarray | Buffer | Sequence[np.ndarray | Buffer]
            A single image (HWC format) or list of images to run the model on.
        ratios : tuple[float, float] | list[tuple[float, float]], optional
            The ratios generated during preprocessing. For single image, pass tuple.
            For batch, pass list.
        padding : tuple[float, float] | list[tuple[float, float]], optional
            The padding values used during preprocessing. For single image, pass tuple.
            For batch, pass list.
        conf_thres : float, optional
            Optional confidence threshold to filter detections
            via during postprocessing. By default None, which will
            use value passed during initialization.
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
            post=self._post_fn(conf_thres, verbose=verbose),
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
    ) -> list[Detection]: ...

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
    ) -> list[list[Detection]]: ...

    def get_detections(
        self: Self,
        outputs: list[np.ndarray] | list[list[np.ndarray]],
        conf_thres: float | None = None,
        nms_iou_thres: float | None = None,
        *,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
        verbose: bool | None = None,
    ) -> list[Detection] | list[list[Detection]]:
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
        nms_iou_thres : float, optional
            The IOU threshold to use during the optional/additional
            NMS operation. By default None, which will use value
            passed during initialization.
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
        list[Detection] | list[list[Detection]]
            For single image: list[Detection] — (bbox, score, class_id) per detection.
            For batch: list[list[Detection]] — detections per image.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["det_get_detections"])

        if verbose:
            LOG.debug(f"{self._tag}: get_detections")

        conf_thres = conf_thres if conf_thres is not None else self._conf_thres
        nms_iou = nms_iou_thres if nms_iou_thres is not None else self._nms_iou
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
        images: np.ndarray | Buffer,
        conf_thres: float | None = ...,
        nms_iou_thres: float | None = ...,
        *,
        extra_nms: bool | None = ...,
        agnostic_nms: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[Detection]: ...

    @overload
    def end2end(
        self: Self,
        images: Sequence[np.ndarray | Buffer],
        conf_thres: float | None = ...,
        nms_iou_thres: float | None = ...,
        *,
        extra_nms: bool | None = ...,
        agnostic_nms: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[list[Detection]]: ...

    def end2end(
        self: Self,
        images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
        conf_thres: float | None = None,
        nms_iou_thres: float | None = None,
        *,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
        verbose: bool | None = None,
    ) -> list[Detection] | list[list[Detection]]:
        """
        Perform end to end inference for a batch of images.

        Equivalent to running preprocess, run, postprocess, and
        get_detections in that order. Makes some memory transfer
        optimizations under the hood to improve performance.

        Parameters
        ----------
        images : np.ndarray | Buffer | Sequence[np.ndarray | Buffer]
            A single image (HWC format) or list of images to perform inference with.
        conf_thres : float, optional
            The confidence threshold with which to retrieve bounding boxes.
            By default None, which will use value passed during initialization.
        nms_iou_thres : float, optional
            The IOU threshold to use during the optional/additional
            NMS operation. By default None, which will use value
            passed during initialization.
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
        list[Detection] | list[list[Detection]]
            For single image: list[Detection] — (bbox, score, class_id) per detection.
            For batch: list[list[Detection]] — detections per image.

        """
        return self._end2end_core(
            images,
            verbose=verbose,
            post=self._post_fn(conf_thres, verbose=verbose),
            get=lambda pp: self.get_detections(
                pp,
                conf_thres=conf_thres,
                nms_iou_thres=nms_iou_thres,
                extra_nms=extra_nms,
                agnostic_nms=agnostic_nms,
                verbose=verbose,
            ),
        )

    def engine_inputs(
        self: Self,
        image: Buffer,
        images: Sequence[np.ndarray | Buffer] | None,
        ratios: list[tuple[float, float]] | None,
        /,
    ) -> list[Buffer]:
        """Build engine inputs, adding per-image orig size / scale factor arrays per input schema."""
        extras: list[Buffer] = []
        if self._use_image_size:
            if images is None or self._input_schema == InputSchema.RT_DETR_V3:
                # the model input size: paddle boxes come back in this canvas, postprocess unletterboxes
                sizes = [(self._input_size[1], self._input_size[0])] * image.shape[0]
            else:
                sizes = [img.shape[:2] for img in images]
            if self._input_schema == InputSchema.RT_DETR:
                # rt-detr scales boxes by (width, height); paddle's im_shape is (height, width)
                sizes = [(w, h) for h, w in sizes]
            extras.append(Buffer.wrap(np.array(sizes, dtype=self._orig_size_dtype)))
        if self._use_scale_factor:
            if ratios is None or self._input_schema == InputSchema.RT_DETR_V3:
                # unscaled: v3 divides by scale_factor, and postprocess undoes the letterbox
                ratios = [(1.0, 1.0)] * image.shape[0]
            extras.append(Buffer.wrap(np.array(ratios, dtype=np.float32)))
        # map by name, as the engine input order can differ from the schema order
        named = dict(zip(self._input_names_without_image(), extras))
        named[self._engine.input_names[self._image_input_index]] = image
        return [named[name] for name in self._engine.input_names]
