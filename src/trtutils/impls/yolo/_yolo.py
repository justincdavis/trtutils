# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from trtutils._engine import TRTEngine

from ._process import get_detections, postprocess, preprocess
from ._version import VALID_VERSIONS

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import Self

_log = logging.getLogger(__name__)


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
        version : int
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
            If input size format is incorrect
            If model does not take 3 channel input

        """
        if version not in VALID_VERSIONS:
            err_msg = f"Invalid version of YOLO given. Received {version}, valid options: {VALID_VERSIONS}"
            raise ValueError(err_msg)
        self._version = version
        self._tag: str = f"{Path(engine_path).stem}-V{self._version}"
        self._engine = TRTEngine(
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            warmup=warmup,
        )
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

    @property
    def engine(self: Self) -> TRTEngine:
        """Get the underlying TRTEngine."""
        return self._engine

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
    ) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
        """
        Preprocess the input.

        Parameters
        ----------
        image : np.ndarray
            The image to preprocess

        Returns
        -------
        tuple[list[np.ndarray], tuple[float, float], tuple[float, float]]
            The preprocessed inputs, rescale ratios, and padding values

        """
        _log.debug(f"{self._tag}: Running preprocess")
        return preprocess(image, self._input_size, self._dtype)

    def postprocess(
        self: Self,
        outputs: list[np.ndarray],
        ratios: tuple[float, float],
        padding: tuple[float, float],
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

        Returns
        -------
        list[np.ndarray]
            The postprocessed outputs

        """
        _log.debug(f"{self._tag}: Running postprocess")
        return postprocess(outputs, self._version, ratios, padding)

    def __call__(
        self: Self,
        image: np.ndarray,
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
    ) -> list[np.ndarray]:
        """
        Run the YOLO network on input.

        Parameters
        ----------
        image : np.ndarray
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
        return self.run(image, preprocessed=preprocessed, postprocess=postprocess)

    def run(
        self: Self,
        image: np.ndarray,
        ratios: tuple[float, float] | None = None,
        padding: tuple[float, float] | None = None,
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
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
        preprocessed : bool, optional
            Whether or not the inputs have been preprocessed.
            If None, will preprocess inputs.
        postprocess : bool, optional
            Whether or not to postprocess the outputs.
            If None, will postprocess outputs.
            If postprocessing will occur and the inputs were
            passed already preprocessed, then the ratios and
            padding must be passed for postprocessing.

        Returns
        -------
        list[np.ndarray]
            The outputs of the YOLO network.

        Raises
        ------
        RuntimeError
            If postprocessing is running, but ratios/padding not found

        """
        if preprocessed is None:
            preprocessed = False
        if postprocess is None:
            postprocess = True

        _log.debug(
            f"{self._tag}: Running: preprocessed: {preprocessed}, postprocess: {postprocess}",
        )

        # handle preprocessing
        if not preprocessed:
            _log.debug("Preprocessing inputs")
            tensor, ratios, padding = self.preprocess(image)
        else:
            tensor = image

        # execute
        outputs = self._engine([tensor])

        # handle postprocessing
        if postprocess:
            _log.debug("Postprocessing outputs")
            if ratios is None or padding is None:
                err_msg = "Must pass ratios/padding if postprocessing and passing already preprocessed inputs."
                raise RuntimeError(err_msg)
            outputs = self.postprocess(outputs, ratios, padding)

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
        conf_thres: float = 0.15,
    ) -> list[tuple[tuple[int, int, int, int], float, int]]:
        """
        Get the bounding boxes of the last output or provided output.

        Parameters
        ----------
        outputs : list[np.ndarray]
            The outputs to process.
        conf_thres : float
            The confidence threshold with which to retrieve bounding boxes.
            By default 0.15

        Returns
        -------
        list[tuple[tuple[int, int, int, int], float, int]]
            The detections

        """
        _log.debug(f"{self._tag}: Running get_detections")
        return get_detections(outputs, version=self._version, conf_thres=conf_thres)
