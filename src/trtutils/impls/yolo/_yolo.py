# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from trtutils._model import TRTModel
from ._process import preprocess, postprocess, get_detections

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import numpy as np
    from typing_extensions import Self


class YOLO:
    """Implementation of YOLO object detectors."""

    def __init__(
        self: Self,
        engine_path: Path | str,
        version: int,
        warmup_iterations: int = 10,
        *,
        warmup: bool | None = None,
    ) -> None:
        """
        Create a YOLO object.

        Parameters
        ----------
        engine_path : Path, str
            The Path or str to the compiled TensorRT engine.
        verison : int
            What version of YOLO the compiled engine is.
            Options are: [7, 8, 9, 10]
        warmup_iterations : int
            The number of warmup iterations to perform.
            The default is 10.
        warmup : bool, optional
            Whether or not to perform warmup iterations.

        Raises
        ------
        ValueError
            If the version number given is not valid

        """
        valid_versions = [7, 8, 9, 10]
        if version not in valid_versions:
            err_msg = f"Invalid version of YOLO given. Received {version}, valid options: {valid_versions}"
            raise ValueError(err_msg)
        self._version = version
        postprocessor: Callable[[list[np.ndarray]], list[np.ndarray]] = partial(
            postprocess, version=self._version
        )
        self._model = TRTModel(
            engine_path=engine_path,
            postprocess=postprocessor,
            warmup_iterations=warmup_iterations,
            warmup=warmup,
        )
        input_spec = self._model._engine.input_spec[0]
        input_size: tuple[int, int] = tuple(input_spec[0])
        dtype = input_spec[1]
        preprocessor: Callable[[list[np.ndarray]], list[np.ndarray]] = partial(
            preprocess,
            input_shape=input_size,
            dtype=dtype,
        )
        self._model.preprocessor = preprocessor

        # storage for last retrived output
        self._last_output: list[np.ndarray] | None = None

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
        return self._model.preprocess(inputs)

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
        return self._model.postprocess(outputs)

    def __call__(
        self: Self,
        inputs: list[np.ndarray],
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
    ) -> list[np.ndarray]:
        """
        Run the YOLO network on input.

        Parameters
        ----------
        inputs : list[np.ndarray]
            The data to run the YOLO network on.
        preprocessed : bool, optional
            Whether or not the inputs have been preprocessed.
            If None, will preprocess inputs.
        postprocess : bool, optional
            Whether or not to postprocess the outputs.
            If None, will postprocess outputs.

        Returns
        -------
        list[np.ndarray]
            The outputs of the YOLO network.

        """
        return self.run(inputs, preprocessed=preprocessed, postprocess=postprocess)

    def run(
        self: Self,
        inputs: list[np.ndarray],
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
    ) -> list[np.ndarray]:
        """
        Run the YOLO network on input.

        Parameters
        ----------
        inputs : list[np.ndarray]
            The data to run the YOLO network on.
        preprocessed : bool, optional
            Whether or not the inputs have been preprocessed.
            If None, will preprocess inputs.
        postprocess : bool, optional
            Whether or not to postprocess the outputs.
            If None, will postprocess outputs.

        Returns
        -------
        list[np.ndarray]
            The outputs of the YOLO network.

        """
        if preprocessed is None:
            preprocessed = False
        if postprocess is None:
            postprocess = True

        outputs = self._model(
            inputs, preprocessed=preprocessed, postprocess=postprocess
        )
        self._last_output = outputs
        return outputs

    def mock_run(
        self: Self,
        inputs: list[np.ndarray] | None = None,
    ) -> list[np.ndarray]:
        """
        Mock an execution of the YOLO model.

        Parameters
        ----------
        inputs : list[np.ndarray], optional
            Optional inputs to use for execution.
            If None, random data will be generated.

        Returns
        -------
        list[np.ndarray]
            The outputs of the model

        """
        return self._model.mock_run(data=inputs)

    def get_detections(
        self: Self,
        outputs: list[np.ndarray] | None = None,
    ) -> list[list[tuple[tuple[int, int, int, int], float, int]]]:
        """
        The the bounding boxes of the last output or provided output.

        Parameters
        ----------
        outputs : list[np.ndarray], optional
            The outputs to process. If None, will use the last outputs of the model.

        Returns
        -------
        list[list[tuple[tuple[int, int, int, int], float, int]]]
            The detections

        Raises
        ------
        ValueError
            If no output is provided and no output has been generated yet.

        """
        if outputs:
            return get_detections(outputs, version=self._version)
        if self._last_output:
            return get_detections(self._last_output, version=self._version)
        err_msg = "No output provided, and no output generated yet."
        raise ValueError(err_msg)
