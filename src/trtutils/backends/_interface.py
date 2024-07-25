# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from typing_extensions import Self


class TRTEngineInterface(ABC):
    @abstractmethod
    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 5,
        *,
        warmup: bool | None = None,
    ) -> None:
        """
        Load the TensorRT engine from a file.

        Parameters
        ----------
        engine_path : Path | str
            The path to the serialized engine file.
        warmup : bool, optional
            Whether to do warmup iterations, by default None
            If None, warmup will be set to False
        warmup_iterations : int, optional
            The number of warmup iterations to do, by default 5

        """

    @abstractmethod
    def __del__(self) -> None:
        """Free the engine resources."""

    @abstractmethod
    def execute(self: Self, data: list[np.ndarray]) -> list[np.ndarray]:
        """
        Execute the engine on the given data.

        Parameters
        ----------
        data : list[np.ndarray]
            The input data.

        Returns
        -------
        list[np.ndarray]
            The output data.

        """

    @property
    @abstractmethod
    def input_spec(self: Self) -> list[tuple[list[int], np.dtype]]:
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.

        Returns
        -------
        list[tuple[list[int], np.dtype]]
            A list with two items per element, the shape and (numpy) datatype of each input tensor.

        """

    @property
    @abstractmethod
    def input_shapes(self: Self) -> list[tuple[int, ...]]:
        """
        Get the shapes for the input tensors of the network.

        Returns
        -------
        list[tuple[int, ...]]
            A list with the shape of each input tensor.

        """

    @property
    @abstractmethod
    def input_dtypes(self: Self) -> list[np.dtype]:
        """
        Get the datatypes for the input tensors of the network.

        Returns
        -------
        list[np.dtype]
            A list with the datatype of each input tensor.

        """

    @property
    @abstractmethod
    def output_spec(self: Self) -> list[tuple[list[int], np.dtype]]:
        """
        Get the specs for the output tensor of the network. Useful to prepare memory allocations.

        Returns
        -------
        list[tuple[list[int], np.dtype]]
            A list with two items per element, the shape and (numpy) datatype of each output tensor.

        """

    @property
    @abstractmethod
    def output_shapes(self: Self) -> list[tuple[int, ...]]:
        """
        Get the shapes for the output tensors of the network.

        Returns
        -------
        list[tuple[int, ...]]
            A list with the shape of each output tensor.

        """

    @property
    @abstractmethod
    def output_dtypes(self: Self) -> list[np.dtype]:
        """
        Get the datatypes for the output tensors of the network.

        Returns
        -------
        list[np.dtype]
            A list with the datatype of each output tensor.

        """

    @cached_property
    def _rng(self: Self) -> np.random.Generator:
        return np.random.default_rng()

    def get_random_input(self: Self) -> list[np.ndarray]:
        """
        Generate a random input for the network.

        Returns
        -------
        list[np.ndarray]
            The random input to the network.

        """
        return [
            self._rng.random(size=shape, dtype=dtype)
            for (shape, dtype) in self.input_spec
        ]

    def __call__(self: Self, data: list[np.ndarray]) -> list[np.ndarray]:
        """
        Execute the network with the given inputs.

        Parameters
        ----------
        data : list[np.ndarray]
            The inputs to the network.

        Returns
        -------
        list[np.ndarray]
            The outputs of the network.

        """
        return self.execute(data)

    def mock_execute(self: Self, data: list[np.ndarray] | None = None) -> None:
        """
        Perform a mock execution of the network.

        This call is useful for warming up the network and
        for testing/benchmarking purposes.

        Parameters
        ----------
        data : list[np.ndarray], optional
            The inputs to the network, by default None
            If None, random inputs will be generated.

        """
        if data is None:
            data = self.get_random_input()
        return self.execute(data)
