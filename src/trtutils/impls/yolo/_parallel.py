# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Thread
from typing import TYPE_CHECKING

import numpy as np

from ._yolo import YOLO

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from typing_extensions import Self

_log = logging.getLogger(__name__)


@dataclass
class _InputPacket:
    data: np.ndarray
    ratio: tuple[float, float] | None = None
    padding: tuple[float, float] | None = None
    preprocessed: bool | None = None
    postprocess: bool | None = None


@dataclass
class _OutputPacket:
    data: list[np.ndarray]
    ratio: tuple[float, float] | None = None
    padding: tuple[float, float] | None = None
    postprocessed: bool | None = None


class ParallelYOLO:
    """
    A parallel implementation of YOLO.

    Allows multiple version of YOLO to be allocated and executed
    at the same time. Primarily useful for multi-gpu/multi-accelerator
    systems such as the NVIDIA Jetson series. Since the TensorRT engines
    are compiled for a specific device, no device specification is needed
    inside of this class.
    """

    def __init__(
        self: Self,
        engines: Sequence[tuple[Path, int]],
        warmup_iterations: int = 100,
        *,
        warmup: bool | None = None,
    ) -> None:
        """
        Create a ParallelYOLO instance.

        Parameters
        ----------
        engines : Sequence[tuple[Path, int]]
            The engine path and version numbers of the YOLO models.
        warmup_iterations : int
            The number of warmup iterations to run.
            Warmup occurs in parallel in each thread.
        warmup : bool, optional
            Whether or not to run the warmup iterations.

        Raises
        ------
        RuntimeError
            If a YOLO model could not be created.

        """
        self._engine_paths = engines
        self._warmup_iterations = warmup_iterations
        self._warmup = warmup
        self._tag = ""
        for epath in self._engine_paths:
            _, eversion = epath
            self._tag += str(eversion)

        self._stopflag = Event()
        self._iqueues: list[Queue[_InputPacket]] = [Queue() for _ in self._engine_paths]
        self._oqueues: list[Queue[_OutputPacket]] = [
            Queue() for _ in self._engine_paths
        ]
        self._profilers: list[tuple[float, float]] = [
            (0.0, 0.0) for _ in self._engine_paths
        ]
        self._flags: list[Event] = [Event() for _ in self._engine_paths]
        self._models: list[YOLO | None] = [None for _ in self._engine_paths]
        self._threads: list[Thread] = [
            Thread(target=self._run, args=(idx,), daemon=True)
            for idx in range(len(self._engine_paths))
        ]
        for thread in self._threads:
            thread.start()
        for flag in self._flags:
            flag.wait()
        for idx, model in enumerate(self._models):
            if model is None:
                self.stop()
                err_msg = f"Error creating YOLO model: {self._engine_paths[idx]}"
                raise RuntimeError(err_msg)

        _log.debug(
            f"{self._tag}: Initialized ParallelYOLO with tag: {self._tag}, num engines: {len(self._models)}",
        )

    def __del__(self: Self) -> None:
        self.stop()
        # if YOLO objects still exist, try manually del them
        with contextlib.suppress(AttributeError):
            for model in self._models:
                with contextlib.suppress(AttributeError):
                    del model

    def get_model(self: Self, modelid: int) -> YOLO:
        """
        Get a YOLO model with id.

        Parameters
        ----------
        modelid : int
            The model ID to get. Based on original list passed during init.

        Returns
        -------
        YOLO
            The YOLO model

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
        inputs: list[np.ndarray],
    ) -> tuple[
        list[np.ndarray],
        list[tuple[float, float]],
        list[tuple[float, float]],
    ]:
        """
        Preprocess inputs for inference.

        Parameters
        ----------
        inputs : list[np.ndarray]
            The inputs to preprocess.

        Returns
        -------
        tuple[list[np.ndarray], list[tuple[float, float]], list[tuple[float, float]]]
            The preprocessed inputs.

        Raises
        ------
        ValueError
            If inputs do not match the number of models

        """
        if len(inputs) != len(self._engine_paths):
            err_msg = "Inputs do not match models"
            raise ValueError(err_msg)

        tensors: list[np.ndarray] = []
        ratios: list[tuple[float, float]] = []
        paddings: list[tuple[float, float]] = []
        for modelid, data in enumerate(inputs):
            tensor, ratio, padding = self.preprocess_model(data, modelid)
            tensors.append(tensor)
            ratios.append(ratio)
            paddings.append(padding)
        return tensors, ratios, paddings

    def preprocess_model(
        self: Self,
        data: np.ndarray,
        modelid: int,
    ) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
        """
        Preprocess data for a specific model.

        Parameters
        ----------
        data : np.ndarray
            The data to preprocess.
        modelid : int
            The model to preprocess the data for.

        Returns
        -------
        tuple[np.ndarray, tuple[float, float], tuple[float, float]]
            The preprocessed data

        """
        _log.debug(f"{self._tag}: Preprocess model: {modelid}")
        return self.get_model(modelid).preprocess(data)

    def postprocess(
        self: Self,
        outputs: list[list[np.ndarray]],
        ratios: list[tuple[float, float]],
        paddings: list[tuple[float, float]],
    ) -> list[list[np.ndarray]]:
        """
        Preprocess outputs for inference.

        Parameters
        ----------
        outputs : list[np.ndarray]
            The outputs to preprocess.
        ratios : list[tuple[float, float]]
            The ratios generated by the preprocess stage.
        paddings : list[tuple[float, float]]
            The paddings generated by the preprocess stage.

        Returns
        -------
        list[list[np.ndarray]]
            The postprocessed outputs.

        Raises
        ------
        ValueError
            If outputs do not match the number of models

        """
        if len(outputs) != len(self._engine_paths):
            err_msg = "Outputs do not match models"
            raise ValueError(err_msg)
        return [
            self.postprocess_model(output, modelid, ratio, padding)
            for modelid, (output, ratio, padding) in enumerate(
                zip(outputs, ratios, paddings),
            )
        ]

    def postprocess_model(
        self: Self,
        outputs: list[np.ndarray],
        modelid: int,
        ratios: tuple[float, float],
        padding: tuple[float, float],
    ) -> list[np.ndarray]:
        """
        Postprocess outputs for a specific model.

        Parameters
        ----------
        outputs : list[np.ndarray]
            The outputs to postprocess.
        modelid : int
            The model to postprocess the data for.
        ratios : tuple[float, float]
            The ratios generated by the preprocess stage
        padding : tuple[float, float]
            The padding generated by the preprocess stage

        Returns
        -------
        tuple[np.ndarray, tuple[float, float], tuple[float, float]]
            The preprocessed data

        """
        _log.debug(f"{self._tag}: Postprocess model: {modelid}")
        return self.get_model(modelid).postprocess(outputs, ratios, padding)

    def get_detections(
        self: Self,
        outputs: list[list[np.ndarray]],
    ) -> list[list[tuple[tuple[int, int, int, int], float, int]]]:
        """
        Get the detections of the YOLO models.

        Parameters
        ----------
        outputs : list[list[np.ndarray]]
            The outputs of the models after postprocessing.

        Returns
        -------
        list[list[tuple[tuple[int, int, int, int], float, int]]]
            The detections

        """
        return [
            self.get_detections_model(output, modelid)
            for modelid, output in enumerate(outputs)
        ]

    def get_detections_model(
        self: Self,
        output: list[np.ndarray],
        modelid: int,
    ) -> list[tuple[tuple[int, int, int, int], float, int]]:
        """
        Get the detections of a single YOLO model.

        Parameters
        ----------
        output : list[np.ndarray]
            The output of a model after postprocessing.
        modelid : int
            The model ID of which model is forming detections.

        Returns
        -------
        list[tuple[tuple[int, int, int, int], float, int]]
            The detections produced by the model

        """
        _log.debug(f"{self._tag}: GetDetections model: {modelid}")
        return self.get_model(modelid).get_detections(output)

    def submit(
        self: Self,
        inputs: list[np.ndarray],
        ratios: list[tuple[float, float]] | None = None,
        paddings: list[tuple[float, float]] | None = None,
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
    ) -> None:
        """
        Submit data to be run for all models or a specific one.

        Parameters
        ----------
        inputs : list[np.ndarray]
            The inputs to pass to the models
        ratios : list[tuple[float, float]], optional
            The optional ratio values for each input
        paddings : list[tuple[float, float]], optional
            The optional padding values for each input
        preprocessed : bool, optional
            Whether or not the inputs are preprocessed
        postprocess : bool, optional
            Whether or not to postprocess the outputs right away

        Raises
        ------
        ValueError
            If the input length does not match the models
            If preprocessed is True, but ratios/paddings not provided

        """
        if len(inputs) != len(self._engine_paths):
            err_msg = "Inputs do not match models."
            raise ValueError(err_msg)
        if (preprocessed and postprocess) and (ratios is None or paddings is None):
            err_msg = "Must provide ratios/paddings if input is marked preprocessed and postprocess is True."
            raise ValueError(err_msg)
        for modelid, data in enumerate(inputs):
            ratio = None if ratios is None else ratios[modelid]
            padding = None if paddings is None else paddings[modelid]
            self.submit_model(
                data,
                modelid,
                ratio,
                padding,
                preprocessed=preprocessed,
                postprocess=postprocess,
            )

    def submit_model(
        self: Self,
        inputs: np.ndarray,
        modelid: int,
        ratio: tuple[float, float] | None = None,
        padding: tuple[float, float] | None = None,
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
    ) -> None:
        """
        Submit data to a specific model.

        Parameters
        ----------
        inputs: np.ndarray
            The data to send to the model.
        modelid : int
            The specific model indix to send the data to.
        ratio : tuple[float, float], optional
            The ratio (if generated) by preprocess.
        padding : tuple[float, float], optional
            The padding (if generated) by postprocess
        preprocessed : bool, optional
            Whether or not the inputs are preprocessed.
        postprocess : bool, optional
            Wherher or not to perform postprocessing.

        """
        packet = _InputPacket(
            data=inputs,
            ratio=ratio,
            padding=padding,
            preprocessed=preprocessed,
            postprocess=postprocess,
        )
        self._iqueues[modelid].put(packet)

    def get_random_input(
        self: Self,
    ) -> list[np.ndarray]:
        """
        Get random inputs.

        Returns
        -------
        list[np.ndarray]
            The random inputs

        """
        return [
            self.get_model(mid).get_random_input() for mid in range(len(self._models))
        ]

    def mock_submit(
        self,
        data: list[np.ndarray] | np.ndarray | None = None,
        modelid: int | None = None,
    ) -> None:
        """
        Perform a mock submit for all models or a specific model.

        Parameters
        ----------
        data : list[np.ndarray], np.ndarray, optional
            The inputs to use for the inference.
        modelid : int, optional
            The specific engine to perform a mock submit for.

        Raises
        ------
        ValueError
            If specified modelid, but gave list of random input
            If gave single np.ndarray input, but did not specify modelid

        """
        if modelid is not None:
            if isinstance(data, np.ndarray):
                self.submit_model(data, modelid, preprocessed=True, postprocess=False)
            elif isinstance(data, list):
                err_msg = "Submitted list[np.ndarray], but specified model ID."
                raise ValueError(err_msg)
            else:
                # need to generate the data
                data = self.get_model(modelid).get_random_input()
                self.submit_model(data, modelid, preprocessed=True, postprocess=False)
        else:
            # the data is a list and no model specified
            if isinstance(data, list):
                self.submit(data, preprocessed=True, postprocess=False)
            elif isinstance(data, np.ndarray):
                err_msg = (
                    "Submitted np.ndarray, but no model ID to specify which model."
                )
                raise ValueError(err_msg)
            else:
                # need to generate the data
                inputs = [
                    self.get_model(mid).get_random_input()
                    for mid in range(len(self._models))
                ]
                self.submit(inputs, preprocessed=True, postprocess=False)

    def retrieve(
        self: Self,
    ) -> tuple[
        list[list[np.ndarray]],
        list[tuple[float, float] | None],
        list[tuple[float, float] | None],
    ]:
        """
        Get outputs back from all the models.

        Returns
        -------
        list[tuple[list[np.ndarray], tuple[float, float], tuple[float, float]]]
            The outputs from all models

        """
        outputs: list[list[np.ndarray]] = []
        ratios: list[tuple[float, float] | None] = []
        paddings: list[tuple[float, float] | None] = []
        for modelid in range(len(self._engine_paths)):
            output, ratio, padding = self.retrieve_model(modelid)
            outputs.append(output)
            ratios.append(ratio)
            paddings.append(padding)
        return outputs, ratios, paddings

    def retrieve_model(
        self: Self,
        modelid: int,
    ) -> tuple[
        list[np.ndarray],
        tuple[float, float] | None,
        tuple[float, float] | None,
    ]:
        """
        Get the outputs from a specific model.

        Parameters
        ----------
        modelid : int
            The model to retrieve data from.

        Returns
        -------
        tuple[list[np.ndarray], tuple[float, float] | None, tuple[float, float] | None]
            The outputs of the model

        """
        packet = self._oqueues[modelid].get()
        return (packet.data, packet.ratio, packet.padding)

    def _run(self: Self, threadid: int) -> None:
        # perform warmup
        engine, version = self._engine_paths[threadid]
        flag = self._flags[threadid]
        try:
            yolo = YOLO(
                engine,
                version=version,
                warmup_iterations=self._warmup_iterations,
                warmup=self._warmup,
            )
        except Exception:
            flag.set()
            raise
        self._models[threadid] = yolo

        # set flag that we are ready
        flag.set()

        # set types
        ratio: tuple[float, float] | None
        padding: tuple[float, float] | None

        # handle inference
        while not self._stopflag.is_set():
            try:
                data = self._iqueues[threadid].get(timeout=0.1)
            except Empty:
                continue
            _log.debug(f"{self._tag}: Received data")

            img = data.data
            if not data.preprocessed:
                img, ratio, padding = yolo.preprocess(img)
            else:
                ratio = data.ratio
                padding = data.padding
            t0 = time.perf_counter()
            results = yolo.run(img, preprocessed=True, postprocess=data.postprocess)
            t1 = time.perf_counter()
            if data.postprocess:
                if ratio is None or padding is None:
                    err_msg = "Ratio/Padding is None, but postprocess set to True."
                    raise ValueError(err_msg)
                results = yolo.postprocess(results, ratio, padding)
            packet = _OutputPacket(
                data=results,
                ratio=ratio,
                padding=padding,
                postprocessed=data.postprocess,
            )
            self._profilers[threadid] = (t0, t1)

            self._oqueues[threadid].put(packet)

        del engine
