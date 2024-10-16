# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import contextlib
from queue import Empty, Queue
from threading import Thread
from typing import TYPE_CHECKING

from ._engine import TRTEngine

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    import numpy as np
    from typing_extensions import Self


def _identity(data: list[np.ndarray]) -> list[np.ndarray]:
    return data


class TRTModel:
    """
    A wrapper around a TensorRT engine that handles the device memory.

    It is thread and process safe to create multiple TRTModels.
    It is valid to create a TRTModel in one thread and use in another.
    Each TRTModel has its own CUDA context and there is no safeguards
    implemented in the class for datarace conditions. As such, a
    single TRTModel should not be used in multiple threads or processes.
    """

    def __init__(
        self: Self,
        engine_path: Path | str,
        preprocess: Callable[[list[np.ndarray]], list[np.ndarray]] = _identity,
        postprocess: Callable[[list[np.ndarray]], list[np.ndarray]] = _identity,
        warmup_iterations: int = 5,
        engine_type: type[TRTEngine] | None = None,
        *,
        warmup: bool | None = None,
    ) -> None:
        """
        Use to initialize the TRTModel.

        Parameters
        ----------
        engine_path : Path, str
            The path to the serialized engine file
        preprocess : callable[[list[np.ndarray]], list[np.ndarray]]
            The function to preprocess the inputs.
            Default is identity function.
        postprocess : callable[[list[np.ndarray]], list[np.ndarray]]
            The function to postprocess the outputs.
            Default is identity function.
        warmup : bool, optional
            Whether to do warmup iterations, by default None
            If None, warmup will be set to False
        warmup_iterations : int, optional
            The number of warmup iterations to do, by default 5
        engine_type : TRTEngine, optional
            An alternative engine type to use, by default None

        """
        trtengine_type: type[TRTEngine] = TRTEngine
        if engine_type is not None:
            trtengine_type = engine_type
        self._engine = trtengine_type(
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            warmup=warmup,
        )
        self._preprocess: Callable[[list[np.ndarray]], list[np.ndarray]] = preprocess
        self._postprocess: Callable[[list[np.ndarray]], list[np.ndarray]] = postprocess

    @property
    def engine(self: Self) -> TRTEngine:
        """Access the underlying TRTEngine."""
        return self._engine

    @property
    def preprocessor(self: Self) -> Callable[[list[np.ndarray]], list[np.ndarray]]:
        """The preprocessing function used in this model."""
        return self._preprocess

    @preprocessor.setter
    def preprocessor(
        self: Self,
        new_preprocess: Callable[[list[np.ndarray]], list[np.ndarray]],
    ) -> None:
        """
        Set the preprocessing function used in this model.

        Useful in case the preprocessor need information which is only
        accessible after loading the engine.
        """
        self._preprocess = new_preprocess

    @property
    def postprocessor(self: Self) -> Callable[[list[np.ndarray]], list[np.ndarray]]:
        """The postprocessing function used in this model."""
        return self._postprocess

    def __call__(
        self: Self,
        inputs: list[np.ndarray],
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
    ) -> list[np.ndarray]:
        """
        Execute the model with the given inputs.

        Parameters
        ----------
        inputs : list[np.ndarray]
            The inputs to the model
        preprocessed : bool, optional
            Whether the inputs are already preprocessed, by default None
            If None, the inputs will be preprocessed
        postprocess : bool, optional
            Whether or not to postprocess the outputs, by default None
            If None, the outputs will be postprocessed

        Returns
        -------
        list[np.ndarray]
            The outputs of the model

        """
        return self.run(inputs, preprocessed=preprocessed, postprocess=postprocess)

    def mock_run(
        self: Self,
        data: list[np.ndarray] | None = None,
    ) -> list[np.ndarray]:
        """
        Execute the model with random inputs.

        Parameters
        ----------
        data : list[np.ndarray], optional
            The inputs to the model, by default None
            If None, random inputs will be used

        Returns
        -------
        list[np.ndarray]
            The outputs of the model

        """
        if data is None:
            data = self._engine.get_random_input()
        return self.run(data, preprocessed=True, postprocess=False)

    def preprocess(self: Self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        """
        Preprocess the inputs.

        Parameters
        ----------
        inputs : list[np.ndarray]
            The inputs to preprocess

        Returns
        -------
        list[np.ndarray]
            The preprocessed inputs

        """
        return self._preprocess(inputs)

    def postprocess(self: Self, outputs: list[np.ndarray]) -> list[np.ndarray]:
        """
        Postprocess the outputs.

        Parameters
        ----------
        outputs : list[np.ndarray]
            The outputs to postprocess

        Returns
        -------
        list[np.ndarray]
            The postprocessed outputs

        """
        return self._postprocess(outputs)

    def run(
        self: Self,
        inputs: list[np.ndarray],
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
    ) -> list[np.ndarray]:
        """
        Execute the model with the given inputs.

        Parameters
        ----------
        inputs : list[np.ndarray]
            The inputs to the model
        preprocessed : bool, optional
            Whether the inputs are already preprocessed, by default None
            If None, the inputs will be preprocessed
        postprocess : bool, optional
            Whether or not to postprocess the outputs, by default None
            If None, the outputs will be postprocessed

        Returns
        -------
        list[np.ndarray]
            The outputs of the model

        """
        if preprocessed is None:
            preprocessed = False
        if postprocess is None:
            postprocess = True
        if not preprocessed:
            inputs = self._preprocess(inputs)
        outputs = self._engine.execute(inputs)
        if postprocess:
            outputs = self._postprocess(outputs)
        return outputs


class QueuedTRTModel:
    """Interact with TRTModel over a Thread and Queue."""

    def __init__(
        self: Self,
        engine_path: Path | str,
        preprocess: Callable[[list[np.ndarray]], list[np.ndarray]] = _identity,
        postprocess: Callable[[list[np.ndarray]], list[np.ndarray]] = _identity,
        warmup_iterations: int = 5,
        engine_type: type[TRTEngine] | None = None,
        *,
        warmup: bool | None = None,
    ) -> None:
        """
        Create a QueuedTRTModel.

        Parameters
        ----------
        engine_path : Path, str
            The Path to the compiled TensorRT Engine.
        preprocess : Callable[[list[np.ndarray]], list[np.ndarray]]
            The function to preprocess the inputs.
            Default is identity function.
        postprocess : Callable[[list[np.ndarray]], list[np.ndarray]]
            The function to postprocess the inputs.
            Default is identity function.
        warmup_iterations : int
            The number of warmup iteratiosn to perform.
            By default 5
        engine_type : type[TRTEngine], optional
            The type of TRTEngine to utilize.
        warmup : bool, optional
            Whether or not to perform the warmup iterations.

        """
        self._stopped = False  # flag for if user stopped thread
        self._model: TRTModel | None = None  # storage for model data
        self._input_queue: Queue[tuple[list[np.ndarray], bool | None]] = Queue()
        self._output_queue: Queue[list[np.ndarray]] = Queue()
        self._thread = Thread(
            target=self._run,
            kwargs={
                "engine_path": engine_path,
                "preprocess": preprocess,
                "postprocess": postprocess,
                "warmup_iterations": warmup_iterations,
                "engine_type": engine_type,
                "warmup": warmup,
            },
        )

    def stop(
        self: Self,
    ) -> None:
        """Stop the thread containing the TRTEngine."""
        self._stopped = True
        self._thread.join()

    def submit(
        self: Self,
        data: list[np.ndarray],
        *,
        preprocessed: bool | None = None,
    ) -> None:
        """
        Put data in the input queue.

        Parameters
        ----------
        data : list[np.ndarray]
            The data to have the engine run.
        preprocessed : bool, optional
            Whether or not the input is already preprocessed.

        """
        self._input_queue.put((data, preprocessed))

    def retrieve(
        self: Self,
        timeout: float | None = None,
    ) -> list[np.ndarray] | None:
        """
        Get an output from the engine thread.

        Parameters
        ----------
        timeout : float, optional
            Timeout for waiting for data.

        Returns
        -------
        list[np.ndarray]
            The output from the engine.

        """
        with contextlib.suppress(Empty):
            return self._output_queue.get(timeout=timeout)
        return None

    def _run(
        self: Self,
        engine_path: Path | str,
        preprocess: Callable[[list[np.ndarray]], list[np.ndarray]],
        postprocess: Callable[[list[np.ndarray]], list[np.ndarray]],
        warmup_iterations: int,
        engine_type: type[TRTEngine] | None,
        *,
        warmup: bool | None = None,
    ) -> None:
        self._model = TRTModel(
            engine_path=engine_path,
            preprocess=preprocess,
            postprocess=postprocess,
            warmup_iterations=warmup_iterations,
            engine_type=engine_type,
            warmup=warmup,
        )

        while not self._stopped:
            try:
                inputs, preprocessed = self._input_queue.get(timeout=0.1)
            except Empty:
                continue

            result = self._model.run(inputs, preprocessed=preprocessed)

            self._output_queue.put(result)


class ParallelTRTModels:
    """Handle many TRTModels in parallel."""

    def __init__(
        self: Self,
        engine_paths: Sequence[Path | str],
        preprocess: Callable[[list[np.ndarray]], list[np.ndarray]]
        | list[Callable[[list[np.ndarray]], list[np.ndarray]]] = _identity,
        postprocess: Callable[[list[np.ndarray]], list[np.ndarray]]
        | list[Callable[[list[np.ndarray]], list[np.ndarray]]] = _identity,
        warmup_iterations: int = 5,
        *,
        warmup: bool | None = None,
    ) -> None:
        """
        Create a ParallelTRTModels instance.

        Parameters
        ----------
        engine_paths : Sequence[Path | str]
            The Paths to the compiled engines to use.
        preprocess : Callable[[list[np.ndarray]], list[np.ndarray]] | list[Callable[[list[np.ndarray]], list[np.ndarray]]]
            The preprocessing function(s)
        postprocess : Callable[[list[np.ndarray]], list[np.ndarray]] | list[Callable[[list[np.ndarray]], list[np.ndarray]]]
            The postprocessing function(s)
        warmup_iterations : int
            The number of iteratiosn to perform warmup for.
            By default 5
        warmup : bool, optional
            Whether or not to run warmup iterations on the engines.

        """
        preprocessors = (
            preprocess
            if isinstance(preprocess, list)
            else [preprocess] * len(engine_paths)
        )
        postprocessors = (
            postprocess
            if isinstance(postprocess, list)
            else [postprocess] * len(engine_paths)
        )
        self._engines: list[QueuedTRTModel] = [
            QueuedTRTModel(
                engine_path=epath,
                preprocess=pre,
                postprocess=post,
                warmup_iterations=warmup_iterations,
                warmup=warmup,
            )
            for epath, pre, post in zip(engine_paths, preprocessors, postprocessors)
        ]

    def stop(self: Self) -> None:
        """Stop the underlying engine threads."""
        for engine in self._engines:
            engine.stop()

    def submit(
        self: Self,
        inputs: list[list[np.ndarray]],
        *,
        preprocessed: bool | None = None,
    ) -> None:
        """
        Submit data to be processed by the engines.

        Parameters
        ----------
        inputs : list[list[np.ndarray]]
            The inputs to pass to the engines.
            Should be a list of the same lenght of engines created.
        preprocessed : bool, optional
            Whether or not the inputs are already preprocessed.

        Raises
        ------
        ValueError
            If the inputs are not the same size as the engines.

        """
        if len(inputs) != len(self._engines):
            err_msg = (
                f"Cannot match {len(inputs)} inputs to {len(self._engines)} engines."
            )
            raise ValueError(err_msg)
        for data, engine in zip(inputs, self._engines):
            engine.submit(data, preprocessed=preprocessed)

    def retrieve(
        self: Self,
        timeout: float | None = None,
    ) -> list[list[np.ndarray] | None]:
        """
        Get the outputs from the engines.

        Parameters
        ----------
        timeout : float, optional
            Timeout for waiting for data.

        Returns
        -------
        list[np.ndarray]
            The output from the engines.

        """
        return [engine.retrieve(timeout=timeout) for engine in self._engines]
