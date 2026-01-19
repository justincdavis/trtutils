# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from trtutils._flags import FLAGS
from trtutils._log import LOG

from ._image_model import ImageModel
from ._schema import InputSchema, OutputSchema, get_detector_io_schema
from .interfaces import DetectorInterface
from .postprocessors import (
    get_detections,
    postprocess_detr,
    postprocess_efficient_nms,
    postprocess_rfdetr,
    postprocess_yolov10,
)
from .preprocessors import CUDAPreprocessor, TRTPreprocessor

if TYPE_CHECKING:
    from pathlib import Path

    from typing_extensions import Self


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
            Only effective with async_v3 backend. Default is None (uses TRTEngine default).
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

        """
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
        self._conf_thres: float = conf_thres
        self._nms_iou: float = nms_iou_thres
        self._nms: bool | None = extra_nms
        self._agnostic_nms: bool | None = agnostic_nms

        # resolve input and output schemas
        self._input_schema: InputSchema
        self._output_schema: OutputSchema

        # auto-detect schemas if needed (function returns both, but we only use what we need)
        auto_input_schema: InputSchema | None = None
        auto_output_schema: OutputSchema | None = None
        if input_schema is None or output_schema is None:
            auto_input_schema, auto_output_schema = get_detector_io_schema(self._engine)

        # resolve input schema
        if input_schema is None:
            self._input_schema = auto_input_schema
        elif isinstance(input_schema, str):
            if input_schema not in InputSchema.names():
                err_msg = f"Invalid input_schema string: {input_schema}. "
                err_msg += f"Valid options: {InputSchema.names()}"
                raise ValueError(err_msg)
            self._input_schema = InputSchema[input_schema]
        else:
            self._input_schema = input_schema

        # resolve output schema
        if output_schema is None:
            self._output_schema = auto_output_schema
        elif isinstance(output_schema, str):
            if output_schema not in OutputSchema.names():
                err_msg = f"Invalid output_schema string: {output_schema}. "
                err_msg += f"Valid options: {OutputSchema.names()}"
                raise ValueError(err_msg)
            self._output_schema = OutputSchema[output_schema]
        else:
            self._output_schema = output_schema

        if self._verbose:
            LOG.debug(f"{self._tag}: Input schema: {self._input_schema}")
            LOG.debug(f"{self._tag}: Output schema: {self._output_schema}")

        # based on the input scheme, we will need to allocate additional attrs
        self._use_image_size: bool = False
        self._use_scale_factor: bool = False
        if self._input_schema == InputSchema.RT_DETR:
            self._use_image_size = True
        elif self._input_schema == InputSchema.RT_DETR_V3:
            self._use_image_size = True
            self._use_scale_factor = True

        # solve for the postprocessing function
        if self._output_schema == OutputSchema.YOLO_V10:
            self._postprocess_fn = postprocess_yolov10
        elif self._output_schema == OutputSchema.RF_DETR:
            self._postprocess_fn = postprocess_rfdetr
        elif self._output_schema == OutputSchema.DETR:
            self._postprocess_fn = postprocess_detr
        else:
            self._postprocess_fn = postprocess_efficient_nms

        # use unified get detections function
        self._get_detections_fn = get_detections

        if self._verbose:
            LOG.debug(f"{self._tag}: Using image size: {self._use_image_size}")
            LOG.debug(f"{self._tag}: Using scale factor: {self._use_scale_factor}")

    @property
    def input_schema(self: Self) -> InputSchema:
        """Get the input schema used by this detector."""
        return self._input_schema

    @property
    def output_schema(self: Self) -> OutputSchema:
        """Get the output schema used by this detector."""
        return self._output_schema

    def preprocess(
        self: Self,
        images: list[np.ndarray],
        resize: str | None = None,
        method: str | None = None,
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]:
        """
        Preprocess the input images.

        Parameters
        ----------
        images : list[np.ndarray]
            The images to preprocess.
        resize : str
            The method to resize the images with.
            Options are [letterbox, linear].
            By default None, which will use the value passed
            during initialization.
        method : str, optional
            The underlying preprocessor to use.
            Options are 'cpu', 'cuda', or 'trt'. By default None, which
            will use the preprocessor stated in the constructor.
        no_copy : bool, optional
            If True and using CUDA, do not copy the
            data from the allocated memory. If the data
            is not copied, it WILL BE OVERWRITTEN INPLACE
            once new data is generated.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]
            The preprocessed batch tensor, list of ratios per image, and list of padding per image.

        """
        resize = resize if resize is not None else self._resize_method
        if verbose:
            LOG.debug(
                f"{self._tag}: Running preprocess, batch_size: {len(images)}, with method: {resize}",
            )
            LOG.debug(f"{self._tag}: Using device: {method}")
        preprocessor = self._preprocessor
        if method is not None:
            if method == "trt" and not FLAGS.TRT_HAS_UINT8:
                method = "cuda"
                LOG.warning(
                    "Preprocessing method set to TensorRT, but platform doesn't support UINT8, fallback to CUDA."
                )
            preprocessor = self._preproc_cpu
            if method == "cuda":
                if self._preproc_cuda is None:
                    self._preproc_cuda = self._setup_cuda_preproc()
                preprocessor = self._preproc_cuda
            elif method == "trt":
                if self._preproc_trt is None:
                    self._preproc_trt = self._setup_trt_preproc()
                preprocessor = self._preproc_trt
        if isinstance(preprocessor, (CUDAPreprocessor, TRTPreprocessor)):
            t0 = time.perf_counter()
            data = preprocessor(images, resize=resize, no_copy=no_copy, verbose=verbose)
            t1 = time.perf_counter()
        else:
            t0 = time.perf_counter()
            data = preprocessor(images, resize=resize, verbose=verbose)
            t1 = time.perf_counter()
        self._pre_profile = (t0, t1)
        return data

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
        return data

    def __call__(
        self: Self,
        images: list[np.ndarray],
        ratios: list[tuple[float, float]] | None = None,
        padding: list[tuple[float, float]] | None = None,
        conf_thres: float | None = None,
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[list[np.ndarray]]:
        """
        Run the model on input.

        Parameters
        ----------
        images : list[np.ndarray]
            The images to run the model on.
        ratios : list[tuple[float, float]], optional
            The ratios generated during preprocessing for each image.
        padding : list[tuple[float, float]], optional
            The padding values used during preprocessing for each image.
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
        list[list[np.ndarray]]
            The postprocessed outputs per image.

        """
        return self.run(
            images,
            ratios,
            padding,
            conf_thres,
            preprocessed=preprocessed,
            postprocess=postprocess,
            no_copy=no_copy,
            verbose=verbose,
        )

    def run(
        self: Self,
        images: list[np.ndarray],
        ratios: list[tuple[float, float]] | None = None,
        padding: list[tuple[float, float]] | None = None,
        conf_thres: float | None = None,
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[list[np.ndarray]]:
        """
        Run the model on input.

        Parameters
        ----------
        images : list[np.ndarray]
            The images to run the model on.
        ratios : list[tuple[float, float]], optional
            The ratios generated during preprocessing for each image.
        padding : list[tuple[float, float]], optional
            The padding values used during preprocessing for each image.
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
        list[list[np.ndarray]]
            The postprocessed outputs per image.

        Raises
        ------
        RuntimeError
            If postprocessing is running, but ratios/padding not found

        """
        if verbose:
            LOG.debug(f"{self._tag}: run")

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
            tensor = images[0] if isinstance(images, list) and len(images) == 1 else images

        batch_size = len(images) if not preprocessed else tensor.shape[0]

        # build input list based on schema
        engine_inputs = [tensor]
        if self._use_image_size:
            # Build batched orig_target_sizes: (batch, 2) with (height, width) per image
            orig_sizes = np.array(
                [img.shape[:2] for img in images]
                if not preprocessed
                else [[self._input_size[1], self._input_size[0]]] * batch_size,
                dtype=np.int32,
            )
            engine_inputs.append(orig_sizes)
        if self._use_scale_factor:
            # Build batched scale_factor: (batch, 2) from ratios list
            scale_factors = np.array(ratios, dtype=np.float32)
            engine_inputs.append(scale_factors)

        # execute
        t0 = time.perf_counter()
        outputs = self._engine(engine_inputs, no_copy=no_copy_run)
        t1 = time.perf_counter()

        # handle postprocessing
        if postprocess:
            if verbose:
                LOG.debug("Postprocessing outputs")
            if ratios is None or padding is None:
                err_msg = "Must pass ratios/padding if postprocessing and passing already preprocessed inputs."
                raise RuntimeError(err_msg)
            outputs = self.postprocess(
                outputs,
                ratios,
                padding,
                conf_thres,
                no_copy=no_copy_post,
            )

        self._infer_profile = (t0, t1)

        return outputs

    def get_detections(
        self: Self,
        outputs: list[list[np.ndarray]],
        conf_thres: float | None = None,
        nms_iou_thres: float | None = None,
        *,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
        verbose: bool | None = None,
    ) -> list[list[tuple[tuple[int, int, int, int], float, int]]]:
        """
        Get the bounding boxes from postprocessed outputs.

        Parameters
        ----------
        outputs : list[list[np.ndarray]]
            The postprocessed outputs per image, each containing
            [bboxes, scores, class_ids].
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
        list[list[tuple[tuple[int, int, int, int], float, int]]]
            The detections per image, where each detection is
            ((x1, y1, x2, y2), score, class_id).

        """
        if verbose:
            LOG.debug(f"{self._tag}: get_detections")

        conf_thres = conf_thres or self._conf_thres
        nms_iou = nms_iou_thres or self._nms_iou
        use_nms = extra_nms if extra_nms is not None else self._nms
        agnostic = agnostic_nms if agnostic_nms is not None else self._agnostic_nms
        return self._get_detections_fn(
            outputs,
            conf_thres=conf_thres,
            nms_iou_thres=nms_iou,
            extra_nms=use_nms,
            agnostic_nms=agnostic,
            verbose=verbose,
        )

    def end2end(
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
        Perform end to end inference for a batch of images.

        Equivalent to running preprocess, run, postprocess, and
        get_detections in that order. Makes some memory transfer
        optimizations under the hood to improve performance.

        Parameters
        ----------
        images : list[np.ndarray]
            The images to perform inference with.
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
        list[list[tuple[tuple[int, int, int, int], float, int]]]
            The detections per image, where each detection is
            ((x1, y1, x2, y2), score, class_id).

        Raises
        ------
        RuntimeError
            If the orig_image_size buffer is not valid
        RuntimeError
            If the scale_factor buffer is not valid

        """
        if verbose:
            LOG.debug(f"{self._tag}: end2end")

        outputs: list[list[np.ndarray]]
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
            input_ptrs = [gpu_ptr]
            if self._use_image_size:
                orig_size_ptr, valid = self._preprocessor.orig_size_allocation
                if valid:
                    input_ptrs.append(orig_size_ptr)
                else:
                    err_msg = "orig_image_size buffer not valid"
                    raise RuntimeError(err_msg)
            if self._use_scale_factor:
                scale_ptr, scale_valid = self._preprocessor.scale_factor_allocation
                if scale_valid:
                    input_ptrs.append(scale_ptr)
                else:
                    err_msg = "scale_factor buffer not valid"
                    raise RuntimeError(err_msg)

            raw_outputs = self._engine.direct_exec(input_ptrs, no_warn=True)
            outputs = self.postprocess(
                raw_outputs,
                ratios,
                padding,
                conf_thres,
                no_copy=True,
                verbose=verbose,
            )

        # generate the detections
        return self.get_detections(
            outputs,
            conf_thres=conf_thres,
            nms_iou_thres=nms_iou_thres,
            extra_nms=extra_nms,
            agnostic_nms=agnostic_nms,
            verbose=verbose,
        )
