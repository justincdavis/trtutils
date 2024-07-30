# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

from ._engine import TRTEngine

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from typing_extensions import Self


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
        engine_path: str,
        preprocess: Callable[[list[np.ndarray]], list[np.ndarray]],
        postprocess: Callable[[list[np.ndarray]], list[np.ndarray]],
        warmup_iterations: int = 5,
        alternative_engine_type: type[TRTEngine] | None = None,
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
        postprocess : callable[[list[np.ndarray]], list[np.ndarray]]
            The function to postprocess the outputs
        warmup : bool, optional
            Whether to do warmup iterations, by default None
            If None, warmup will be set to False
        warmup_iterations : int, optional
            The number of warmup iterations to do, by default 5
        alternative_engine_type : TRTEngineInterface, optional
            An alternative engine type to use, by default None

        """
        engine_type: type[TRTEngine] = TRTEngine
        if alternative_engine_type is not None:
            engine_type = alternative_engine_type
        self._engine = engine_type(
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            warmup=warmup,
        )
        self._preprocess = preprocess
        self._postprocess = postprocess

    def __call__(
        self: Self,
        inputs: list[np.ndarray],
        *,
        preprocessed: bool | None = None,
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


        Returns
        -------
        list[np.ndarray]
            The outputs of the model

        """
        return self.run(inputs, preprocessed=preprocessed)

    def mock_run(
        self: Self,
        data: list[np.ndarray] | None = None,
        *,
        preprocessed: bool | None = None,
    ) -> list[np.ndarray]:
        """
        Execute the model with random inputs.

        Parameters
        ----------
        data : list[np.ndarray], optional
            The inputs to the model, by default None
            If None, random inputs will be used
        preprocessed : bool, optional
            Whether the inputs are already preprocessed, by default None
            If None, the inputs will be preprocessed.

        Returns
        -------
        list[np.ndarray]
            The outputs of the model

        """
        if data is None:
            data = self._engine.get_random_input()
        outputs = self.run(data, preprocessed=preprocessed)
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


        Returns
        -------
        list[np.ndarray]
            The outputs of the model

        """
        if preprocessed is None:
            preprocessed = False
        if not preprocessed:
            inputs = self._preprocess(inputs)
        outputs = self._engine.execute(inputs)
        return self._postprocess(outputs)
