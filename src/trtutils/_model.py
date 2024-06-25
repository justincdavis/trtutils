# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ._engine import TRTEngine

if TYPE_CHECKING:
    from typing_extensions import Self


class TRTModel:
    """
    A wrapper around a TensorRT engine that handles the device memory.

    It is thread and process safe to create multiple TRTModels.
    It is valid to create a TRTModel in one thread and use in another.
    Each TRTModel has its own CUDA context and there is no safeguards
    implemented in the class for datarace conditions. As such, a
    single TRTModel should not be used in multiple threads or processes.


    Attributes
    ----------
    input_shapes : list[tuple[int, ...]]
        The shapes of the inputs


    Methods
    -------
    __call__(inputs: list[np.ndarray], preprocessed: bool | None = None)
        Execute the model with the given inputs
    preprocess(inputs: list[np.ndarray])
        Preprocess the inputs
    run(inputs: list[np.ndarray], preprocessed: bool | None = None)
        Execute the model with the given inputs
    mock_run()
        Execute the model with random inputs

    """

    def __init__(
        self: Self,
        engine_path: str,
        preprocess: Callable[[list[np.ndarray]], list[np.ndarray]],
        postprocess: Callable[[list[np.ndarray]], Any],
        warmup_iterations: int = 5,
        dtype: np.number = np.float32,  # type: ignore[assignment]
        device: int = 0,
        *,
        warmup: bool | None = None,
    ) -> None:
        """
        Use to initialize the TRTModel.

        Parameters
        ----------
        engine_path : str
            The path to the serialized engine file
        preprocess : callable[[list[np.ndarray]], list[np.ndarray]]
            The function to preprocess the inputs
        postprocess : callable[[list[np.ndarray]], Any]
            The function to postprocess the outputs
        warmup : bool, optional
            Whether to do warmup iterations, by default None
            If None, warmup will be set to False
        warmup_iterations : int, optional
            The number of warmup iterations to do, by default 5
        dtype : np.number, optional
            The datatype to use for the inputs and outputs, by default np.float32
        device : int, optional
            The device to use, by default 0

        """
        self._engine = TRTEngine(
            engine_path,
            warmup_iterations,
            dtype,
            device,
            warmup=warmup,
        )
        self._preprocess = preprocess
        self._postprocess = postprocess

    def __call__(
        self: Self,
        inputs: list[np.ndarray],
        *,
        preprocessed: bool | None = None,
    ) -> Any:  # noqa: ANN401
        """
        Execute the model with the given inputs.

        Parameters
        ----------
        inputs : list[np.ndarray]
            The inputs to the model
        preprocessed : bool, optional
            Whether the inputs are already preprocessed, by default None
            If None, the inputs will be preprocessed


        Returns
        -------
        Any
            The outputs of the model

        """
        return self.run(inputs, preprocessed=preprocessed)

    def mock_run(self: Self) -> Any:  # noqa: ANN401
        """
        Execute the model with random inputs.

        Returns
        -------
        Any
            The outputs of the model

        """
        outputs = self._engine.mock_execute()
        return self._postprocess(outputs)

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

    def run(
        self: Self,
        inputs: list[np.ndarray],
        *,
        preprocessed: bool | None = None,
    ) -> Any:  # noqa: ANN401
        """
        Execute the model with the given inputs.

        Parameters
        ----------
        inputs : list[np.ndarray]
            The inputs to the model
        preprocessed : bool, optional
            Whether the inputs are already preprocessed, by default None
            If None, the inputs will be preprocessed


        Returns
        -------
        Any
            The outputs of the model

        """
        if preprocessed is None:
            preprocessed = False
        if not preprocessed:
            inputs = self._preprocess(inputs)
        outputs = self._engine.execute(inputs)
        return self._postprocess(outputs)
