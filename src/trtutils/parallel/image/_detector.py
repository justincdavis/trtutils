# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Thread
from typing import TYPE_CHECKING, cast

import numpy as np
import nvtx
from typing_extensions import TypeGuard

from trtutils._flags import FLAGS
from trtutils._log import LOG
from trtutils.image._detector import Detector

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from typing_extensions import Self


def _is_raw_outputs(
    outputs: list[np.ndarray] | list[list[np.ndarray]],
) -> TypeGuard[list[np.ndarray]]:
    return not outputs or isinstance(outputs[0], np.ndarray)


def _is_postprocessed_batches(
    outputs: list[list[list[np.ndarray]] | list[np.ndarray]],
) -> TypeGuard[list[list[list[np.ndarray]]]]:
    return all(
        isinstance(model_outputs, list) and (not model_outputs or isinstance(model_outputs[0], list))
        for model_outputs in outputs
    )


@dataclass
class _InputPacket:
    data: list[np.ndarray]
    ratios: list[tuple[float, float]] | None = None
    padding: list[tuple[float, float]] | None = None
    preprocess_method: str | None = "trt"
    preprocessed: bool | None = None
    postprocess: bool | None = None
    no_copy: bool | None = None


@dataclass
class _OutputPacket:
    data: list[np.ndarray] | list[list[np.ndarray]]
    ratios: list[tuple[float, float]] | None = None
    padding: list[tuple[float, float]] | None = None
    postprocessed: bool | None = None


@dataclass
class EngineInfo:
    """
    Configuration for a single engine in ParallelDetector.

    All fields except engine_path are optional. When None, the value from
    ParallelDetector's constructor will be used as the default.
    """

    engine_path: Path | str
    detector_class: type[Detector] = Detector
    # Per-engine overrides (None = use ParallelDetector defaults)
    dla_core: int | None = None
    input_range: tuple[float, float] | None = None
    preprocessor: str | None = None
    resize_method: str | None = None
    conf_thres: float | None = None
    nms_iou_thres: float | None = None
    mean: tuple[float, float, float] | None = None
    std: tuple[float, float, float] | None = None
    input_schema: str | None = None
    output_schema: str | None = None
    backend: str | None = None
    warmup: bool | None = None
    pagelocked_mem: bool | None = None
    unified_mem: bool | None = None
    cuda_graph: bool | None = None
    extra_nms: bool | None = None
    agnostic_nms: bool | None = None


class ParallelDetector:
    """
    A parallel implementation of Detector.

    Allows multiple version of Detector to be allocated and executed
    at the same time. Primarily useful for multi-gpu/multi-accelerator
    systems such as the NVIDIA Jetson series. Since the TensorRT engines
    are compiled for a specific device, no device specification is needed
    inside of this class.
    """

    def __init__(
        self: Self,
        engines: Sequence[EngineInfo],
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0.0, 1.0),
        preprocessor: str = "trt",
        resize_method: str = "letterbox",
        conf_thres: float = 0.1,
        nms_iou_thres: float = 0.5,
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = False,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
        sequential_load: bool | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        """
        Create a ParallelDetector instance.

        Parameters
        ----------
        engines : Sequence[EngineInfo]
            The information required for creation of the Detector models.
            Per-engine options in EngineInfo override the defaults specified here.
        warmup_iterations : int
            The number of warmup iterations to run.
            Warmup occurs in parallel in each thread. Default is 10.
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
            By default 0.1.
        nms_iou_thres : float
            The IOU threshold to use in the optional and additional
            NMS operation. By default 0.5.
        mean : tuple[float, float, float] | None
            The mean values to use for the imagenet normalization.
            By default None, which means no normalization will be applied.
        std : tuple[float, float, float] | None
            The standard deviation values to use for the imagenet normalization.
            By default None, which means no normalization will be applied.
        backend : str
            The execution backend to use. Options are ['auto', 'async_v3', 'async_v2'].
            Default is 'auto', which selects the best available backend.
        warmup : bool, optional
            Whether or not to run the warmup iterations.
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
            end-to-end execution in the end2end() method.
            Only effective with async_v3 backend. Default is True.
        extra_nms : bool, optional
            Whether or not an additional CPU-side NMS operation
            should be conducted on final detections.
        agnostic_nms : bool, optional
            Whether or not the optional/additional NMS operation
            should perform class agnostic NMS.
        sequential_load : bool, optional
            If True, load models one at a time instead of in parallel.
            This is useful when memory is constrained or when hardware
            resources (e.g., DLA cores) cannot be initialized concurrently.
            Default is None (parallel loading).
        no_warn : bool, optional
            If True, suppresses warnings from TensorRT during engine deserialization.
            Default is None, which means warnings will be shown.
        verbose : bool, optional
            Whether or not to output additional information.
            Only covers initialization of the models.

        Raises
        ------
        RuntimeError
            If a Detector model could not be created.

        """
        self._engine_info: list[EngineInfo] = list(engines)
        self._tag = str(len(self._engine_info))
        self._nvtx_tags = {
            "init": f"parallel_detector::init [{self._tag}]",
            "preprocess": f"parallel_detector::preprocess [{self._tag}]",
            "preprocess_model": f"parallel_detector::preprocess_model [{self._tag}]",
            "postprocess": f"parallel_detector::postprocess [{self._tag}]",
            "postprocess_model": f"parallel_detector::postprocess_model [{self._tag}]",
            "get_detections": f"parallel_detector::get_detections [{self._tag}]",
            "get_detections_model": f"parallel_detector::get_detections_model [{self._tag}]",
            "submit": f"parallel_detector::submit [{self._tag}]",
            "submit_model": f"parallel_detector::submit_model [{self._tag}]",
            "retrieve": f"parallel_detector::retrieve [{self._tag}]",
            "retrieve_model": f"parallel_detector::retrieve_model [{self._tag}]",
            "end2end": f"parallel_detector::end2end [{self._tag}]",
        }
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["init"])
        self._warmup_iterations = warmup_iterations
        self._input_range = input_range
        self._preprocessor = preprocessor
        self._resize_method = resize_method
        self._conf_thres = conf_thres
        self._nms_iou_thres = nms_iou_thres
        self._mean = mean
        self._std = std
        self._backend = backend
        self._warmup = warmup
        self._pagelocked_mem = pagelocked_mem
        self._unified_mem = unified_mem
        self._cuda_graph = cuda_graph
        self._extra_nms = extra_nms
        self._agnostic_nms = agnostic_nms
        self._sequential_load = sequential_load
        self._no_warn = no_warn
        self._verbose = verbose

        self._stopflag = Event()
        self._iqueues: list[Queue[_InputPacket]] = [Queue() for _ in self._engine_info]
        self._oqueues: list[Queue[_OutputPacket]] = [Queue() for _ in self._engine_info]
        self._profilers: list[tuple[float, float]] = [(0.0, 0.0) for _ in self._engine_info]
        self._flags: list[Event] = [Event() for _ in self._engine_info]
        self._models: list[Detector | None] = [None for _ in self._engine_info]
        self._threads: list[Thread] = [
            Thread(target=self._run, args=(idx,), daemon=True)
            for idx in range(len(self._engine_info))
        ]
        if self._sequential_load:
            # Load models one at a time
            for idx, thread in enumerate(self._threads):
                thread.start()
                self._flags[idx].wait()
        else:
            # Load models in parallel (default)
            for thread in self._threads:
                thread.start()
            for flag in self._flags:
                flag.wait()
        for idx, model in enumerate(self._models):
            if model is None:
                self.stop()
                err_msg = f"Error creating Detector model: {self._engine_info[idx].engine_path}"
                if FLAGS.NVTX_ENABLED:
                    nvtx.pop_range()  # init
                raise RuntimeError(err_msg)

        if self._verbose:
            LOG.debug(
                f"{self._tag}: Initialized ParallelDetector with tag: {self._tag}, num engines: {len(self._models)}",
            )
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # init

    def __del__(self: Self) -> None:
        self.stop()
        # if Detector objects still exist, try manually del them
        with contextlib.suppress(AttributeError):
            for model in self._models:
                with contextlib.suppress(AttributeError):
                    del model

    @property
    def models(self: Self) -> list[Detector]:
        """
        Get the underlying Detector models.

        Returns
        -------
        list[Detector]
            A list of the underlying models.

        Raises
        ------
        RuntimeError
            If any values are None, not initialized yet.

        """
        models = [m for m in self._models.copy() if m is not None]
        if len(models) != len(self._models):
            err_msg = "Some or all models are None, models are initalized yet."
            raise RuntimeError(err_msg)
        return models

    def get_model(self: Self, modelid: int) -> Detector:
        """
        Get a Detector model with id.

        Parameters
        ----------
        modelid : int
            The model ID to get. Based on original list passed during init.

        Returns
        -------
        Detector
            The Detector model

        Raises
        ------
        RuntimeError
            If access is attempted before init is complete

        """
        model = self._models[modelid]
        if model is None:
            err_msg = "Accessing models before intialization completed."
            raise RuntimeError(err_msg)
        return model

    def get_model_profiling(self: Self, modelid: int) -> tuple[float, float, float]:
        """
        Get the latency of a specific model as profiled in thread.

        Parameters
        ----------
        modelid : int
            The model ID of the profiling to get.

        Returns
        -------
        tuple[float, float, float]
            The start time, end time, and delta

        """
        t0, t1 = self._profilers[modelid]
        return t0, t1, t1 - t0

    def get_profiling(self: Self) -> list[tuple[float, float, float]]:
        """
        Get all the profiling results for all models.

        Returns
        -------
        list[tuple[float, float, float]]
            The profiling data

        """
        return [self.get_model_profiling(idx) for idx in range(len(self._models))]

    def stop(self: Self) -> None:
        """Stop the threads."""
        self._stopflag.set()
        with contextlib.suppress(RuntimeError, TimeoutError):
            for thread in self._threads:
                thread.join()

    def preprocess(
        self: Self,
        inputs: list[list[np.ndarray]],
        resize: str = "letterbox",
        method: str | None = None,
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> tuple[
        list[np.ndarray],
        list[list[tuple[float, float]]],
        list[list[tuple[float, float]]],
    ]:
        """
        Preprocess inputs for inference.

        Parameters
        ----------
        inputs : list[list[np.ndarray]]
            The inputs to preprocess, one batch per model.
        resize : str
            The method to resize the image with.
            By default letterbox, options are [letterbox, linear]
        method : str, optional
            The underlying preprocessor to use.
            Options are 'cpu' and 'cuda'. By default None, which
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
        tuple[list[np.ndarray], list[list[tuple[float, float]]], list[list[tuple[float, float]]]]
            The preprocessed tensors, ratios per image per model, and padding per image per model.

        Raises
        ------
        ValueError
            If inputs do not match the number of models

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["preprocess"])
        if len(inputs) != len(self._engine_info):
            err_msg = "Inputs do not match models"
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()
            raise ValueError(err_msg)

        tensors: list[np.ndarray] = []
        ratios: list[list[tuple[float, float]]] = []
        paddings: list[list[tuple[float, float]]] = []
        for modelid, batch in enumerate(inputs):
            tensor, ratio_list, padding_list = self.preprocess_model(
                batch,
                modelid,
                resize=resize,
                method=method,
                no_copy=no_copy,
                verbose=verbose,
            )
            tensors.append(tensor)
            ratios.append(ratio_list)
            paddings.append(padding_list)
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()
        return tensors, ratios, paddings

    def preprocess_model(
        self: Self,
        images: list[np.ndarray],
        modelid: int,
        resize: str = "letterbox",
        method: str | None = None,
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]:
        """
        Preprocess a batch of images for a specific model.

        Parameters
        ----------
        images : list[np.ndarray]
            The batch of images to preprocess.
        modelid : int
            The model to preprocess the data for.
        resize : str
            The method to resize the image with.
            By default letterbox, options are [letterbox, linear]
        method : str, optional
            The underlying preprocessor to use.
            Options are 'cpu' and 'cuda'. By default None, which
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
            The preprocessed tensor, ratios per image, and padding per image.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["preprocess_model"])
        if verbose:
            LOG.debug(f"{self._tag}: Preprocess model: {modelid}")
        result = self.get_model(modelid).preprocess(
            images,
            resize=resize,
            method=method,
            no_copy=no_copy,
            verbose=verbose,
        )
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()
        return result

    def postprocess(
        self: Self,
        outputs: list[list[np.ndarray]],
        ratios: list[list[tuple[float, float]]],
        paddings: list[list[tuple[float, float]]],
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[list[list[np.ndarray]]]:
        """
        Postprocess outputs for inference.

        Parameters
        ----------
        outputs : list[list[np.ndarray]]
            The raw outputs per model.
        ratios : list[list[tuple[float, float]]]
            The ratios per image per model.
        paddings : list[list[tuple[float, float]]]
            The paddings per image per model.
        no_copy : bool, optional
            If True, do not copy the data from the allocated
            memory. If the data is not copied, it WILL BE
            OVERWRITTEN INPLACE once new data is generated.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[list[list[np.ndarray]]]
            The postprocessed outputs per image per model.

        Raises
        ------
        ValueError
            If outputs do not match the number of models

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["postprocess"])
        if len(outputs) != len(self._engine_info):
            err_msg = "Outputs do not match models"
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()
            raise ValueError(err_msg)
        result = [
            self.postprocess_model(
                output, modelid, ratio_list, padding_list, no_copy=no_copy, verbose=verbose
            )
            for modelid, (output, ratio_list, padding_list) in enumerate(
                zip(outputs, ratios, paddings),
            )
        ]
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()
        return result

    def postprocess_model(
        self: Self,
        outputs: list[np.ndarray],
        modelid: int,
        ratios: list[tuple[float, float]],
        padding: list[tuple[float, float]],
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[list[np.ndarray]]:
        """
        Postprocess outputs for a specific model.

        Parameters
        ----------
        outputs : list[np.ndarray]
            The raw outputs to postprocess.
        modelid : int
            The model to postprocess the data for.
        ratios : list[tuple[float, float]]
            The ratios per image from preprocessing.
        padding : list[tuple[float, float]]
            The padding per image from preprocessing.
        no_copy : bool, optional
            If True, do not copy the data from the allocated
            memory. If the data is not copied, it WILL BE
            OVERWRITTEN INPLACE once new data is generated.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[list[np.ndarray]]
            The postprocessed outputs per image.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["postprocess_model"])
        if verbose:
            LOG.debug(f"{self._tag}: Postprocess model: {modelid}")
        result = self.get_model(modelid).postprocess(
            outputs,
            ratios,
            padding,
            no_copy=no_copy,
            verbose=verbose,
        )
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()
        return result

    def get_detections(
        self: Self,
        outputs: list[list[list[np.ndarray]]],
        *,
        verbose: bool | None = None,
    ) -> list[list[list[tuple[tuple[int, int, int, int], float, int]]]]:
        """
        Get the detections of the YOLO models.

        Parameters
        ----------
        outputs : list[list[list[np.ndarray]]]
            The postprocessed outputs per image per model.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[list[list[tuple[tuple[int, int, int, int], float, int]]]]
            The detections per image per model.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["get_detections"])
        result = [
            self.get_detections_model(output, modelid, verbose=verbose)
            for modelid, output in enumerate(outputs)
        ]
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()
        return result

    def get_detections_model(
        self: Self,
        outputs: list[list[np.ndarray]],
        modelid: int,
        *,
        verbose: bool | None = None,
    ) -> list[list[tuple[tuple[int, int, int, int], float, int]]]:
        """
        Get the detections for a batch from a single model.

        Parameters
        ----------
        outputs : list[list[np.ndarray]]
            The postprocessed outputs per image.
        modelid : int
            The model ID of which model is forming detections.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[list[tuple[tuple[int, int, int, int], float, int]]]
            The detections per image.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["get_detections_model"])
        if verbose:
            LOG.debug(f"{self._tag}: GetDetections model: {modelid}")
        result = self.get_model(modelid).get_detections(outputs, verbose=verbose)
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()
        return result

    def submit(
        self: Self,
        inputs: list[list[np.ndarray]],
        ratios: list[list[tuple[float, float]]] | None = None,
        paddings: list[list[tuple[float, float]]] | None = None,
        preprocess_method: str | None = None,
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        """
        Submit batches to all models.

        Parameters
        ----------
        inputs : list[list[np.ndarray]]
            The batches to pass to each model.
        ratios : list[list[tuple[float, float]]], optional
            The ratios per image per model.
        paddings : list[list[tuple[float, float]]], optional
            The padding per image per model.
        preprocess_method : str, optional
            The method to use for preprocessing.
            Options are 'cpu', 'cuda', 'trt'. By default None, which
            will use the preprocessor stated in the constructor.
        preprocessed : bool, optional
            Whether or not the inputs are preprocessed
        postprocess : bool, optional
            Whether or not to postprocess the outputs right away
        no_copy : bool, optional
            If True, do not copy the data from the allocated
            memory. If the data is not copied, it WILL BE
            OVERWRITTEN INPLACE once new data is generated.
        verbose : bool, optional
            Whether or not to log additional information.

        Raises
        ------
        ValueError
            If the input length does not match the models
            If preprocessed is True, but ratios/paddings not provided

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["submit"])
        if len(inputs) != len(self._engine_info):
            err_msg = "Inputs do not match models."
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()
            raise ValueError(err_msg)
        if (preprocessed and postprocess) and (ratios is None or paddings is None):
            err_msg = "Must provide ratios/paddings if input is marked preprocessed and postprocess is True."
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()
            raise ValueError(err_msg)
        for modelid, batch in enumerate(inputs):
            ratio_list = None if ratios is None else ratios[modelid]
            padding_list = None if paddings is None else paddings[modelid]
            self.submit_model(
                batch,
                modelid,
                ratio_list,
                padding_list,
                preprocess_method=preprocess_method,
                preprocessed=preprocessed,
                postprocess=postprocess,
                no_copy=no_copy,
                verbose=verbose,
            )
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()

    def submit_model(
        self: Self,
        images: list[np.ndarray],
        modelid: int,
        ratios: list[tuple[float, float]] | None = None,
        padding: list[tuple[float, float]] | None = None,
        preprocess_method: str | None = None,
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        """
        Submit a batch to a specific model.

        Parameters
        ----------
        images : list[np.ndarray]
            The batch of images to send to the model.
        modelid : int
            The specific model index to send the data to.
        ratios : list[tuple[float, float]], optional
            The ratios per image from preprocessing.
        padding : list[tuple[float, float]], optional
            The padding per image from preprocessing.
        preprocess_method : str, optional
            The method to use for preprocessing.
            Options are 'cpu', 'cuda', 'trt'. By default None, which
            will use the preprocessor stated in the constructor.
        preprocessed : bool, optional
            Whether or not the inputs are preprocessed.
        postprocess : bool, optional
            Whether or not to perform postprocessing.
        no_copy : bool, optional
            If True, do not copy the data from the allocated
            memory. If the data is not copied, it WILL BE
            OVERWRITTEN INPLACE once new data is generated.
        verbose : bool, optional
            Whether or not to log additional information.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["submit_model"])
        if verbose:
            LOG.debug(f"{self._tag}: Submit model: {modelid}")
        packet = _InputPacket(
            data=images,
            ratios=ratios,
            padding=padding,
            preprocess_method=preprocess_method,
            preprocessed=preprocessed,
            postprocess=postprocess,
            no_copy=no_copy,
        )
        self._iqueues[modelid].put(packet)
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()

    def get_random_input(
        self: Self,
    ) -> list[list[np.ndarray]]:
        """
        Get random inputs (one per model).

        Returns
        -------
        list[list[np.ndarray]]
            The random inputs per model.

        """
        return [self.get_model(mid).get_random_input() for mid in range(len(self._models))]

    def mock_submit(
        self,
        data: list[list[np.ndarray]] | list[np.ndarray] | None = None,
        modelid: int | None = None,
    ) -> None:
        """
        Perform a mock submit for all models or a specific model.

        Parameters
        ----------
        data : list[list[np.ndarray]], list[np.ndarray], optional
            The inputs to use for the inference. If modelid is specified,
            should be list[np.ndarray] (single batch). Otherwise should be
            list[list[np.ndarray]] (batch per model).
        modelid : int, optional
            The specific engine to perform a mock submit for.

        Raises
        ------
        ValueError
            If specified modelid, but gave list of batches
            If gave single batch, but did not specify modelid

        """
        if modelid is not None:
            if data is not None and len(data) > 0 and isinstance(data[0], np.ndarray):
                # Single batch for single model
                # Type narrowing: data is list[np.ndarray] here
                data_single: list[np.ndarray] = cast("list[np.ndarray]", data)
                self.submit_model(
                    data_single,
                    modelid,
                    preprocessed=True,
                    postprocess=False,
                    no_copy=True,
                )
            elif data is not None and len(data) > 0 and isinstance(data[0], list):
                err_msg = "Submitted list[list[np.ndarray]], but specified model ID."
                raise ValueError(err_msg)
            else:
                # Generate random data
                random_input = self.get_model(modelid).get_random_input()
                self.submit_model(
                    random_input,
                    modelid,
                    preprocessed=True,
                    postprocess=False,
                    no_copy=True,
                )
        else:
            if data is not None and len(data) > 0 and isinstance(data[0], list):
                # Batches for all models
                # Type narrowing: data is list[list[np.ndarray]] here
                data_batches: list[list[np.ndarray]] = cast("list[list[np.ndarray]]", data)
                self.submit(data_batches, preprocessed=True, postprocess=False, no_copy=True)
            elif data is not None and len(data) > 0 and isinstance(data[0], np.ndarray):
                err_msg = "Submitted list[np.ndarray], but no model ID to specify which model."
                raise ValueError(err_msg)
            else:
                # Generate random data
                inputs = [self.get_model(mid).get_random_input() for mid in range(len(self._models))]
                self.submit(inputs, preprocessed=True, postprocess=False, no_copy=True)

    def retrieve(
        self: Self,
        *,
        verbose: bool | None = None,
    ) -> tuple[
        list[list[list[np.ndarray]] | list[np.ndarray]],
        list[list[tuple[float, float]] | None],
        list[list[tuple[float, float]] | None],
    ]:
        """
        Get outputs back from all the models.

        Parameters
        ----------
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        tuple[
            list[list[list[np.ndarray]] | list[np.ndarray]],
            list[list[tuple[float, float]] | None],
            list[list[tuple[float, float]] | None],
        ]
            The outputs per image per model, ratios per image per model, padding per image per model.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["retrieve"])
        outputs: list[list[list[np.ndarray]] | list[np.ndarray]] = []
        ratios: list[list[tuple[float, float]] | None] = []
        paddings: list[list[tuple[float, float]] | None] = []
        for modelid in range(len(self._engine_info)):
            output, ratio_list, padding_list = self.retrieve_model(modelid, verbose=verbose)
            outputs.append(output)
            ratios.append(ratio_list)
            paddings.append(padding_list)
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()
        return outputs, ratios, paddings

    def retrieve_model(
        self: Self,
        modelid: int,
        *,
        verbose: bool | None = None,
    ) -> tuple[
        list[list[np.ndarray]] | list[np.ndarray],
        list[tuple[float, float]] | None,
        list[tuple[float, float]] | None,
    ]:
        """
        Get the outputs from a specific model.

        Parameters
        ----------
        modelid : int
            The model to retrieve data from.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        tuple[list[list[np.ndarray]] | list[np.ndarray], list[tuple[float, float]] | None, list[tuple[float, float]] | None]
            The outputs per image, ratios per image, and padding per image.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["retrieve_model"])
        if verbose:
            LOG.debug(f"{self._tag}: Retrieve model: {modelid}")
        packet = self._oqueues[modelid].get()
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()
        return (packet.data, packet.ratios, packet.padding)

    def end2end(
        self: Self,
        inputs: list[list[np.ndarray]],
        ratios: list[list[tuple[float, float]]] | None = None,
        paddings: list[list[tuple[float, float]]] | None = None,
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[list[list[tuple[tuple[int, int, int, int], float, int]]]]:
        """
        Perform end-to-end inference for all models.

        Parameters
        ----------
        inputs : list[list[np.ndarray]]
            The batches to pass to each model.
        ratios : list[list[tuple[float, float]]], optional
            The ratios per image per model.
        paddings : list[list[tuple[float, float]]], optional
            The padding per image per model.
        preprocessed : bool, optional
            Whether or not the inputs are preprocessed
        postprocess : bool, optional
            Whether or not to postprocess the outputs right away
        no_copy : bool, optional
            If True, do not copy the data from the allocated
            memory. If the data is not copied, it WILL BE
            OVERWRITTEN INPLACE once new data is generated.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[list[list[tuple[tuple[int, int, int, int], float, int]]]]
            The detections per image per model.

        Raises
        ------
        ValueError
            If postprocess is False when calling end2end.
        RuntimeError
            If postprocessed outputs are not available for end2end.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["end2end"])
        if postprocess is None:
            postprocess = True
        if not postprocess:
            err_msg = "end2end requires postprocess to be True."
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()
            raise ValueError(err_msg)
        self.submit(
            inputs,
            ratios,
            paddings,
            preprocessed=preprocessed,
            postprocess=postprocess,
            no_copy=no_copy,
            verbose=verbose,
        )
        outputs, _, _ = self.retrieve(verbose=verbose)
        if not _is_postprocessed_batches(outputs):
            err_msg = "Expected postprocessed outputs for end2end."
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()
            raise RuntimeError(err_msg)
        result = self.get_detections(outputs, verbose=verbose)
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()
        return result

    def _run(self: Self, threadid: int) -> None:
        # perform warmup
        info = self._engine_info[threadid]
        det_class = info.detector_class
        flag = self._flags[threadid]

        # Resolve per-engine overrides vs global defaults
        input_range = info.input_range if info.input_range is not None else self._input_range
        preprocessor = info.preprocessor if info.preprocessor is not None else self._preprocessor
        resize_method = info.resize_method if info.resize_method is not None else self._resize_method
        conf_thres = info.conf_thres if info.conf_thres is not None else self._conf_thres
        nms_iou_thres = info.nms_iou_thres if info.nms_iou_thres is not None else self._nms_iou_thres
        mean = info.mean if info.mean is not None else self._mean
        std = info.std if info.std is not None else self._std
        backend = info.backend if info.backend is not None else self._backend
        warmup = info.warmup if info.warmup is not None else self._warmup
        pagelocked_mem = (
            info.pagelocked_mem if info.pagelocked_mem is not None else self._pagelocked_mem
        )
        unified_mem = info.unified_mem if info.unified_mem is not None else self._unified_mem
        cuda_graph = info.cuda_graph if info.cuda_graph is not None else self._cuda_graph
        extra_nms = info.extra_nms if info.extra_nms is not None else self._extra_nms
        agnostic_nms = info.agnostic_nms if info.agnostic_nms is not None else self._agnostic_nms

        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(f"parallel_detector::_run::model_init [{self._tag}:{threadid}]")
        try:
            detector = det_class(
                engine_path=info.engine_path,
                warmup_iterations=self._warmup_iterations,
                input_range=input_range,
                preprocessor=preprocessor,
                resize_method=resize_method,
                conf_thres=conf_thres,
                nms_iou_thres=nms_iou_thres,
                mean=mean,
                std=std,
                input_schema=info.input_schema,
                output_schema=info.output_schema,
                dla_core=info.dla_core,
                backend=backend,
                warmup=warmup,
                pagelocked_mem=pagelocked_mem,
                unified_mem=unified_mem,
                cuda_graph=cuda_graph,
                extra_nms=extra_nms,
                agnostic_nms=agnostic_nms,
                no_warn=self._no_warn,
                verbose=self._verbose,
            )
        except Exception:
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()  # model_init
            flag.set()
            raise
        self._models[threadid] = detector
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # model_init

        # set flag that we are ready
        flag.set()

        # set types
        ratios: list[tuple[float, float]] | None
        padding: list[tuple[float, float]] | None

        # handle inference
        while not self._stopflag.is_set():
            try:
                data = self._iqueues[threadid].get(timeout=0.1)
            except Empty:
                continue
            LOG.debug(f"{self._tag}: Received data")
            if FLAGS.NVTX_ENABLED:
                nvtx.push_range(f"parallel_detector::_run [{self._tag}:{threadid}]")

            images = data.data

            # path 1: GPU preprocess needed, make end2end optimizations
            if not data.preprocessed and (
                data.preprocess_method == "cuda" or data.preprocess_method == "trt"
            ):
                if FLAGS.NVTX_ENABLED:
                    nvtx.push_range(
                        f"parallel_detector::_run::gpu_preprocess [{self._tag}:{threadid}]"
                    )
                preproc = (
                    detector._preproc_trt  # noqa: SLF001
                    if data.preprocess_method == "trt"
                    else detector._preproc_cuda  # noqa: SLF001
                )
                # if the preprocessor is None, need to use preprocess method to create it
                if preproc is None:
                    tensor, ratios, padding = detector.preprocess(
                        images,
                        method=data.preprocess_method,
                        no_copy=data.no_copy,
                    )
                    t0 = time.perf_counter()
                    # Run inference only (postprocess=False), handle postprocessing separately
                    results = detector.run(
                        [tensor],
                        ratios=ratios,
                        padding=padding,
                        preprocessed=True,
                        postprocess=False,
                        no_copy=data.no_copy,
                    )
                    t1 = time.perf_counter()
                else:
                    # direct_preproc handles batch
                    gpu_img, ratios, padding = preproc.direct_preproc(images, no_warn=True)
                    t0 = time.perf_counter()
                    results = detector.engine.direct_exec([gpu_img], no_warn=True)
                    t1 = time.perf_counter()
                if FLAGS.NVTX_ENABLED:
                    nvtx.pop_range()  # gpu_preprocess

            # path 2: preprocess not needed, or CPU preprocessing
            else:
                if FLAGS.NVTX_ENABLED:
                    nvtx.push_range(
                        f"parallel_detector::_run::cpu_preprocess [{self._tag}:{threadid}]"
                    )
                if not data.preprocessed:
                    tensor, ratios, padding = detector.preprocess(
                        images,
                        method=data.preprocess_method,
                        no_copy=data.no_copy,
                    )
                else:
                    if len(images) != 1:
                        err_msg = (
                            "Preprocessed inputs must be a list containing a single batch tensor."
                        )
                        if FLAGS.NVTX_ENABLED:
                            nvtx.pop_range()  # cpu_preprocess
                            nvtx.pop_range()  # _run
                        raise ValueError(err_msg)
                    tensor = images[0]
                    ratios = data.ratios
                    padding = data.padding
                t0 = time.perf_counter()
                # Run inference only (postprocess=False), handle postprocessing separately
                results = detector.run(
                    [tensor],
                    ratios=ratios,
                    padding=padding,
                    preprocessed=True,
                    postprocess=False,
                    no_copy=data.no_copy,
                )
                t1 = time.perf_counter()
                if FLAGS.NVTX_ENABLED:
                    nvtx.pop_range()  # cpu_preprocess

            # run the postprocessing (common for all paths)
            if FLAGS.NVTX_ENABLED:
                nvtx.push_range(f"parallel_detector::_run::postprocess [{self._tag}:{threadid}]")
            if data.postprocess:
                if ratios is None or padding is None:
                    err_msg = "Ratios/Padding is None, but postprocess set to True."
                    if FLAGS.NVTX_ENABLED:
                        nvtx.pop_range()  # postprocess
                        nvtx.pop_range()  # _run
                    raise ValueError(err_msg)
                if not _is_raw_outputs(results):
                    err_msg = "Expected raw detector outputs before postprocess."
                    if FLAGS.NVTX_ENABLED:
                        nvtx.pop_range()  # postprocess
                        nvtx.pop_range()  # _run
                    raise RuntimeError(err_msg)
                postproc_results = detector.postprocess(
                    results,
                    ratios,
                    padding,
                    no_copy=data.no_copy,
                )
            else:
                postproc_results = results
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()  # postprocess

            packet = _OutputPacket(
                data=postproc_results,
                ratios=ratios,
                padding=padding,
                postprocessed=data.postprocess,
            )
            self._profilers[threadid] = (t0, t1)

            self._oqueues[threadid].put(packet)
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()  # _run

        del detector
