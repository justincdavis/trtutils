# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from trtutils._engine import TRTEngine
from trtutils._flags import FLAGS
from trtutils._log import LOG

from ._preprocessors import CPUPreprocessor, CUDAPreprocessor, TRTPreprocessor
from ._process import get_detections, postprocess

if TYPE_CHECKING:
    from typing_extensions import Self


class YOLO:
    """Implementation of YOLO object detectors."""

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0.0, 1.0),
        preprocessor: str = "trt",
        resize_method: str = "letterbox",
        conf_thres: float = 0.1,
        nms_iou_thres: float = 0.5,
        dla_core: int | None = None,
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        """
        Create a YOLO object.

        Parameters
        ----------
        engine_path : Path, str
            The Path or str to the compiled TensorRT engine.
        version : int
            What version of YOLO the compiled engine is.
            Options are: [7, 8, 9, 10]
        warmup_iterations : int
            The number of warmup iterations to perform.
            The default is 10.
        input_range : tuple[float, float]
            The range of input values which should be passed to
            the model. By default [0.0, 1.0].
            Versions 7/8/9/10 expect 0.0 through 1.0
            X expects 0.0 through 255.0
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
        dla_core : int, optional
            The DLA core to assign DLA layers of the engine to. Default is None.
            If None, any DLA layers will be assigned to DLA core 0.
        warmup : bool, optional
            Whether or not to perform warmup iterations.
        pagelocked_mem : bool, optional
            Whether or not to use pagelocked memory for underlying CUDA operations.
            By default, pagelocked memory will be used.
        unified_mem : bool, optional
            Whether or not the system has unified memory.
            If True, use cudaHostAllocMapped to take advantage of unified memory.
            By default None, which means the default host allocation will be used.
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
            If the version number given is not valid
            If input size format is incorrect
            If model does not take 3 channel input

        """
        self._tag: str = f"{Path(engine_path).stem}"
        if verbose:
            LOG.debug(f"Creating YOLO: {self._tag}")

        self._pagelocked_mem = pagelocked_mem if pagelocked_mem is not None else True
        self._engine = TRTEngine(
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            warmup=warmup,
            dla_core=dla_core,
            pagelocked_mem=self._pagelocked_mem,
            unified_mem=unified_mem,
            no_warn=no_warn,
        )
        self._unified_mem = self._engine.unified_mem
        self._conf_thres = conf_thres
        self._resize_method: str = resize_method
        self._nms_iou: float = nms_iou_thres
        self._nms: bool | None = extra_nms
        self._agnostic_nms: bool | None = agnostic_nms
        input_spec = self._engine.input_spec[0]
        input_size: tuple[int, ...] = tuple(input_spec[0])
        yolo_input_size = 4
        if len(input_size) != yolo_input_size:
            err_msg = "Expected YOLO model to have input size of form: (batch, channels, height, width)"
            err_msg += f", found {input_size}"
            raise ValueError(err_msg)
        rgb_channels = 3
        if input_size[1] != rgb_channels:
            err_msg = f"Expected YOLO model to take {rgb_channels} channel input, found {input_size[1]}"
            raise ValueError(err_msg)
        self._input_size: tuple[int, int] = (input_size[3], input_size[2])
        self._dtype = input_spec[1]
        self._input_range = input_range

        # set up the preprocessor
        self._preprocessor: CPUPreprocessor | CUDAPreprocessor | TRTPreprocessor
        valid_preprocessors = ["cpu", "cuda", "trt"]
        if preprocessor not in valid_preprocessors:
            err_msg = f"Invalid preprocessor found, options are: {valid_preprocessors}"
            raise ValueError(err_msg)
        self._preproc_cpu: CPUPreprocessor = CPUPreprocessor(
            self._input_size,
            self._input_range,
            self._dtype,
            tag=self._tag,
        )
        self._preproc_cuda: CUDAPreprocessor | None = None
        self._preproc_trt: TRTPreprocessor | None = None
        # change the preprocessor setup to cuda if set to trt and trt doesnt have uint8 support
        if preprocessor == "trt" and not FLAGS.TRT_HAS_UINT8:
            preprocessor = "cuda"
            LOG.warning(
                "Preprocessing method set to TensorRT, but platform doesnt have UINT8 support, fallback to CUDA."
            )
        # existing logic
        if preprocessor == "trt":
            self._preproc_trt = self._setup_trt_preproc()
            self._preprocessor = self._preproc_trt
        elif preprocessor == "cuda" and self._dtype == np.float32:
            self._preproc_cuda = self._setup_cuda_preproc()
            self._preprocessor = self._preproc_cuda
        else:
            self._preprocessor = self._preproc_cpu

        # basic profiler setup
        self._pre_profile: tuple[float, float] = (0.0, 0.0)
        self._infer_profile: tuple[float, float] = (0.0, 0.0)
        self._post_profile: tuple[float, float] = (0.0, 0.0)

        # if warmup, warmup the preprocessors
        if warmup:
            self._preprocessor.warmup()

    def _setup_cuda_preproc(self: Self) -> CUDAPreprocessor:
        return CUDAPreprocessor(
            self._input_size,
            self._input_range,
            self._dtype,
            resize=self._resize_method,
            stream=self._engine.stream,
            pagelocked_mem=self._pagelocked_mem,
            unified_mem=self._unified_mem,
            tag=self._tag,
        )

    def _setup_trt_preproc(self: Self) -> TRTPreprocessor:
        return TRTPreprocessor(
            self._input_size,
            self._input_range,
            self._dtype,
            resize=self._resize_method,
            stream=self._engine.stream,
            pagelocked_mem=self._pagelocked_mem,
            unified_mem=self._unified_mem,
            tag=self._tag,
        )

    @property
    def engine(self: Self) -> TRTEngine:
        """Get the underlying TRTEngine."""
        return self._engine

    @property
    def name(self: Self) -> str:
        """Get the name of the engine."""
        return self._engine.name

    @property
    def input_shape(self: Self) -> tuple[int, int]:
        """Get the width, height input shape."""
        return self._input_size

    @property
    def dtype(self: Self) -> np.dtype:
        """Get the dtype required by the model."""
        return self._dtype

    def preprocess(
        self: Self,
        image: np.ndarray,
        resize: str | None = None,
        method: str | None = None,
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
        """
        Preprocess the input.

        Parameters
        ----------
        image : np.ndarray
            The image to preprocess
        resize : str
            The method to resize the image with.
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
        tuple[list[np.ndarray], tuple[float, float], tuple[float, float]]
            The preprocessed inputs, rescale ratios, and padding values

        """
        resize = resize if resize is not None else self._resize_method
        if verbose:
            LOG.debug(
                f"{self._tag}: Running preprocess, shape: {image.shape}, with method: {resize}",
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
            data = preprocessor(image, resize=resize, no_copy=no_copy, verbose=verbose)
            t1 = time.perf_counter()
        else:
            t0 = time.perf_counter()
            data = preprocessor(image, resize=resize, verbose=verbose)
            t1 = time.perf_counter()
        self._pre_profile = (t0, t1)
        return data

    def postprocess(
        self: Self,
        outputs: list[np.ndarray],
        ratios: tuple[float, float],
        padding: tuple[float, float],
        conf_thres: float | None = None,
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[np.ndarray]:
        """
        Postprocess the outputs.

        Parameters
        ----------
        outputs : list[np.ndarray]
            The outputs to postprocess
        ratios : tuple[float, float]
            The rescale ratios used during preprocessing
        padding : tuple[float, float]
            The padding values used during preprocessing
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
        list[np.ndarray]
            The postprocessed outputs

        """
        if verbose:
            LOG.debug(f"{self._tag}: postprocess")

        conf_thres = conf_thres or self._conf_thres
        t0 = time.perf_counter()
        data = postprocess(outputs, ratios, padding, conf_thres, no_copy=no_copy)
        t1 = time.perf_counter()
        self._post_profile = (t0, t1)
        return data

    def __call__(
        self: Self,
        image: np.ndarray,
        ratios: tuple[float, float] | None = None,
        padding: tuple[float, float] | None = None,
        conf_thres: float | None = None,
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[np.ndarray]:
        """
        Run the YOLO network on input.

        Parameters
        ----------
        image : np.ndarray
            The data to run the YOLO network on.
        ratios : tuple[float, float], optional
            The ratios generated during preprocessing.
        padding : tuple[float, float], optional
            The padding values used during preprocessing.
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
        list[np.ndarray]
            The outputs of the YOLO network.

        """
        return self.run(
            image,
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
        image: np.ndarray,
        ratios: tuple[float, float] | None = None,
        padding: tuple[float, float] | None = None,
        conf_thres: float | None = None,
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[np.ndarray]:
        """
        Run the YOLO network on input.

        Parameters
        ----------
        image: np.ndarray
            The data to run the YOLO network on.
        ratios : tuple[float, float], optional
            The ratios generated during preprocessing.
        padding : tuple[float, float], optional
            The padding values used during preprocessing.
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
        list[np.ndarray]
            The outputs of the YOLO network.

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
            tensor, ratios, padding = self.preprocess(image, no_copy=no_copy_pre)
        else:
            tensor = image

        # execute
        t0 = time.perf_counter()
        outputs = self._engine([tensor], no_copy=no_copy_run)
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

    def get_random_input(
        self: Self,
    ) -> np.ndarray:
        """
        Generate a random image for the YOLO model.

        Returns
        -------
        np.ndarray
            The random image.

        """
        return self._engine.get_random_input()[0]

    def mock_run(
        self: Self,
        image: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """
        Mock an execution of the YOLO model.

        Parameters
        ----------
        image : np.ndarray, optional
            Optional inputs to use for execution.
            If None, random data will be generated.

        Returns
        -------
        list[np.ndarray]
            The outputs of the model

        """
        if image is not None:
            return self._engine.mock_execute(data=[image])
        return self._engine.mock_execute()

    def get_detections(
        self: Self,
        outputs: list[np.ndarray],
        conf_thres: float | None = None,
        nms_iou_thres: float | None = None,
        *,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
        verbose: bool | None = None,
    ) -> list[tuple[tuple[int, int, int, int], float, int]]:
        """
        Get the bounding boxes of the last output or provided output.

        Parameters
        ----------
        outputs : list[np.ndarray]
            The outputs to process.
        conf_thres : float, optional
            The confidence threshold with which to retrieve bounding boxes.
            By default None, which will use value passed during initialization
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
        list[tuple[tuple[int, int, int, int], float, int]]
            The detections

        """
        if verbose:
            LOG.debug(f"{self._tag}: get_detections")

        conf_thres = conf_thres or self._conf_thres
        nms_iou = nms_iou_thres or self._nms_iou
        use_nms = extra_nms if extra_nms is not None else self._nms
        agnostic = agnostic_nms if agnostic_nms is not None else self._agnostic_nms
        return get_detections(
            outputs,
            conf_thres=conf_thres,
            nms_iou_thres=nms_iou,
            extra_nms=use_nms,
            agnostic_nms=agnostic,
            verbose=verbose,
        )

    def end2end(
        self: Self,
        image: np.ndarray,
        conf_thres: float | None = None,
        nms_iou_thres: float | None = None,
        *,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
        verbose: bool | None = None,
    ) -> list[tuple[tuple[int, int, int, int], float, int]]:
        """
        Perform end to end inference for a YOLO model.

        Equivalent to running preprocess, run, postprocess, and
        get_detections in that order. Makes some memory transfer
        optimizations under the hood to improve performance.

        Parameters
        ----------
        image : np.ndarray
            The image to perform inference with.
        conf_thres : float, optional
            The confidence threshold with which to retrieve bounding boxes.
            By default None
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
        list[tuple[tuple[int, int, int, int], float, int]]
            The detections where each entry is bbox, conf, class_id

        """
        if verbose:
            LOG.debug(f"{self._tag}: end2end")

        outputs: list[np.ndarray]
        # if using CPU preprocessor best you can do is remove host-to-host copies
        if not isinstance(self._preprocessor, (CUDAPreprocessor, TRTPreprocessor)):
            if verbose:
                LOG.debug(f"{self._tag}: end2end -> calling CPU preprocess")

            outputs = self.run(
                image,
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
                image,
                resize=self._resize_method,
                no_warn=True,
                verbose=verbose,
            )
            outputs = self._engine.direct_exec([gpu_ptr], no_warn=True)
            outputs = self.postprocess(
                outputs,
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
