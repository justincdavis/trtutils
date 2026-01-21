# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Thread
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import TypeGuard

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
    engine_path: Path | str
    dla_core: int | None = None
    detector_class: type[Detector] = Detector


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
        warmup_iterations: int = 100,
        *,
        warmup: bool | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        """
        Create a ParallelDetector instance.

        Parameters
        ----------
        engines : Sequence[EngineInfo]
            The information required for creation of the Detector models.
        warmup_iterations : int
            The number of warmup iterations to run.
            Warmup occurs in parallel in each thread.
        warmup : bool, optional
            Whether or not to run the warmup iterations.
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
        self._warmup_iterations = warmup_iterations
        self._warmup = warmup
        self._tag = str(len(self._engine_info))
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
        for thread in self._threads:
            thread.start()
        for flag in self._flags:
            flag.wait()
        for idx, model in enumerate(self._models):
            if model is None:
                self.stop()
                err_msg = f"Error creating Detector model: {self._engine_info[idx].engine_path}"
                raise RuntimeError(err_msg)

        if self._verbose:
            LOG.debug(
                f"{self._tag}: Initialized ParallelDetector with tag: {self._tag}, num engines: {len(self._models)}",
            )

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
        models = self._models.copy()
        if not all(models):
            err_msg = "Some or all models are None, models are initalized yet."
            raise RuntimeError(err_msg)
        return models  # type: ignore[return-value]

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
        if len(inputs) != len(self._engine_info):
            err_msg = "Inputs do not match models"
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
        if verbose:
            LOG.debug(f"{self._tag}: Preprocess model: {modelid}")
        return self.get_model(modelid).preprocess(
            images,
            resize=resize,
            method=method,
            no_copy=no_copy,
            verbose=verbose,
        )

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
        if len(outputs) != len(self._engine_info):
            err_msg = "Outputs do not match models"
            raise ValueError(err_msg)
        return [
            self.postprocess_model(
                output, modelid, ratio_list, padding_list, no_copy=no_copy, verbose=verbose
            )
            for modelid, (output, ratio_list, padding_list) in enumerate(
                zip(outputs, ratios, paddings),
            )
        ]

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
        if verbose:
            LOG.debug(f"{self._tag}: Postprocess model: {modelid}")
        return self.get_model(modelid).postprocess(
            outputs,
            ratios,
            padding,
            no_copy=no_copy,
            verbose=verbose,
        )

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
        return [
            self.get_detections_model(output, modelid, verbose=verbose)
            for modelid, output in enumerate(outputs)
        ]

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
        if verbose:
            LOG.debug(f"{self._tag}: GetDetections model: {modelid}")
        return self.get_model(modelid).get_detections(outputs, verbose=verbose)

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
        if len(inputs) != len(self._engine_info):
            err_msg = "Inputs do not match models."
            raise ValueError(err_msg)
        if (preprocessed and postprocess) and (ratios is None or paddings is None):
            err_msg = "Must provide ratios/paddings if input is marked preprocessed and postprocess is True."
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
                self.submit_model(
                    data,  # type: ignore[arg-type]
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
                self.submit(data, preprocessed=True, postprocess=False, no_copy=True)  # type: ignore[arg-type]
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
        outputs: list[list[list[np.ndarray]] | list[np.ndarray]] = []
        ratios: list[list[tuple[float, float]] | None] = []
        paddings: list[list[tuple[float, float]] | None] = []
        for modelid in range(len(self._engine_info)):
            output, ratio_list, padding_list = self.retrieve_model(modelid, verbose=verbose)
            outputs.append(output)
            ratios.append(ratio_list)
            paddings.append(padding_list)
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
        if verbose:
            LOG.debug(f"{self._tag}: Retrieve model: {modelid}")
        packet = self._oqueues[modelid].get()
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
        if postprocess is None:
            postprocess = True
        if not postprocess:
            err_msg = "end2end requires postprocess to be True."
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
            raise RuntimeError(err_msg)
        return self.get_detections(outputs, verbose=verbose)

    def _run(self: Self, threadid: int) -> None:
        # perform warmup
        det_class = self._engine_info[threadid].detector_class
        flag = self._flags[threadid]
        try:
            detector = det_class(
                engine_path=self._engine_info[threadid].engine_path,
                warmup_iterations=self._warmup_iterations,
                warmup=self._warmup,
                dla_core=self._engine_info[threadid].dla_core,
                no_warn=self._no_warn,
                verbose=self._verbose,
            )
        except Exception:
            flag.set()
            raise
        self._models[threadid] = detector

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

            images = data.data

            # path 1: GPU preprocess needed, make end2end optimizations
            if not data.preprocessed and (
                data.preprocess_method == "cuda" or data.preprocess_method == "trt"
            ):
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

            # path 2: preprocess not needed, or CPU preprocessing
            else:
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

            # run the postprocessing (common for all paths)
            if data.postprocess:
                if ratios is None or padding is None:
                    err_msg = "Ratios/Padding is None, but postprocess set to True."
                    raise ValueError(err_msg)
                if not _is_raw_outputs(results):
                    err_msg = "Expected raw detector outputs before postprocess."
                    raise RuntimeError(err_msg)
                postproc_results = detector.postprocess(
                    results,
                    ratios,
                    padding,
                    no_copy=data.no_copy,
                )
            else:
                postproc_results = results  # type: ignore[assignment]

            packet = _OutputPacket(
                data=postproc_results,
                ratios=ratios,
                padding=padding,
                postprocessed=data.postprocess,
            )
            self._profilers[threadid] = (t0, t1)

            self._oqueues[threadid].put(packet)

        del detector
