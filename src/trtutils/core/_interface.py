# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ._bindings import allocate_bindings
from ._engine import create_engine

if TYPE_CHECKING:
    from typing_extensions import Self


class TRTEngineInterface(ABC):
    def __init__(
        self: Self,
        engine_path: Path | str,
    ) -> None:
        """
        Load the TensorRT engine from a file.

        Parameters
        ----------
        engine_path : Path | str
            The path to the serialized engine file.

        """
        # store path stem as name
        self._name = Path(engine_path).stem

        # engine, context, logger, and CUDA stream
        self._engine, self._context, self._logger, self._stream = create_engine(
            engine_path,
        )

        # allocate memory for inputs and outputs
        self._inputs, self._outputs, self._allocations, self._batch_size = (
            allocate_bindings(
                self._engine,
                self._context,
            )
        )
        self._input_allocations: list[int] = [
            input_b.allocation for input_b in self._inputs
        ]
        self._output_allocations: list[int] = [
            output_b.allocation for output_b in self._outputs
        ]

        # store cache random data
        self._rand_input: list[np.ndarray] | None = None

    @property
    def name(self: Self) -> str:
        """The name of the engine, as the stem of the Path."""
        return self._name

    def __del__(self: Self) -> None:
        def _del(obj: object, attr: str) -> None:
            with contextlib.suppress(AttributeError):
                delattr(obj, attr)

        with contextlib.suppress(AttributeError):
            for binding in self._inputs:
                with contextlib.suppress(RuntimeError):
                    binding.free()
        with contextlib.suppress(AttributeError):
            for binding in self._outputs:
                with contextlib.suppress(RuntimeError):
                    binding.free()

        attrs = ["_context", "_engine"]
        for attr in attrs:
            _del(self, attr)

    @abstractmethod
    def execute(
        self: Self,
        data: list[np.ndarray],
        *,
        no_copy: bool | None = None,
    ) -> list[np.ndarray]:
        """
        Execute the network with the given inputs.

        Parameters
        ----------
        data : list[np.ndarray]
            The inputs to the network.
        no_copy : bool, optional
            If True, the outputs will not be copied out
            from the cuda allocated host memory. Instead,
            the host memory will be returned directly.
            This memory WILL BE OVERWRITTEN INPLACE
            by future inferences.

        Returns
        -------
        list[np.ndarray]
            The outputs of the network.

        """

    @abstractmethod
    def direct_exec(
        self: Self,
        pointers: list[int],
        *,
        no_warn: bool | None = None,
    ) -> list[np.ndarray]:
        """
        Execute the network with the given GPU memory pointers.

        The outputs of this function are not copied on return.
        The data will be updated inplace if execute or direct_exec
        is called. Calling this method while giving bad pointers
        will also cause CUDA runtime to crash and program to crash.

        Parameters
        ----------
        pointers : list[int]
            The inputs to the network.
        no_warn : bool, optional
            If True, do not warn about usage.

        Returns
        -------
        list[np.ndarray]
            The outputs of the network.

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

    def get_random_input(self: Self, *, new: bool | None = None) -> list[np.ndarray]:
        """
        Generate a random input for the network.

        Parameters
        ----------
        new : bool, optional
            Whether or not to generate new input. By default None/False.

        Returns
        -------
        list[np.ndarray]
            The random input to the network.

        """
        if new or self._rand_input is None:
            rand_input = [
                self._rng.random(size=shape, dtype=dtype)
                for (shape, dtype) in self.input_spec
            ]
            self._rand_input = rand_input
            return self._rand_input
        return self._rand_input

    def __call__(
        self: Self,
        data: list[np.ndarray],
        *,
        no_copy: bool | None = None,
    ) -> list[np.ndarray]:
        """
        Execute the network with the given inputs.

        Parameters
        ----------
        data : list[np.ndarray]
            The inputs to the network.
        no_copy : bool, optional
            If True, the outputs will not be copied out
            from the cuda allocated host memory. Instead,
            the host memory will be returned directly.
            This memory WILL BE OVERWRITTEN INPLACE
            by future inferences.

        Returns
        -------
        list[np.ndarray]
            The outputs of the network.

        """
        return self.execute(data, no_copy=no_copy)

    def mock_execute(
        self: Self,
        data: list[np.ndarray] | None = None,
    ) -> list[np.ndarray]:
        """
        Perform a mock execution of the network.

        This call is useful for warming up the network and
        for testing/benchmarking purposes.

        Parameters
        ----------
        data : list[np.ndarray], optional
            The inputs to the network, by default None
            If None, random inputs will be generated.

        Returns
        -------
        list[np.ndarray]
            The outputs of the network.

        """
        if data is None:
            data = self.get_random_input()
        return self.execute(data)

    def warmup(self: Self, iterations: int) -> None:
        """
        Warmup the network for a given number of iterations.

        Parameters
        ----------
        iterations : int
            The number of iterations to warmup the network.

        """
        for _ in range(iterations):
            self.mock_execute()
