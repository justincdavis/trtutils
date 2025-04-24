# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from trtutils._flags import FLAGS

from ._bindings import allocate_bindings
from ._engine import create_engine
from ._stream import stream_synchronize

if TYPE_CHECKING:
    from typing_extensions import Self

    with contextlib.suppress(ImportError):
        import tensorrt as trt

        try:
            import cuda.bindings.runtime as cudart
        except (ImportError, ModuleNotFoundError):
            from cuda import cudart


class TRTEngineInterface(ABC):
    def __init__(
        self: Self,
        engine_path: Path | str,
        *,
        pagelocked_mem: bool | None = None,
        no_warn: bool | None = None,
    ) -> None:
        """
        Load the TensorRT engine from a file.

        Parameters
        ----------
        engine_path : Path | str
            The path to the serialized engine file.
        pagelocked_mem : bool, optional
            Whether or not to use pagelocked memory for host allocations.
            By default None, which means pagelocked memory will be used.
        no_warn : bool, optional
            If True, suppresses warnings from TensorRT during engine deserialization.
            Default is None, which means warnings will be shown.

        """
        # store path stem as name
        self._name = Path(engine_path).stem

        # engine, context, logger, and CUDA stream
        self._engine, self._context, self._logger, self._stream = create_engine(
            engine_path,
            no_warn=no_warn,
        )

        # allocate memory for inputs and outputs
        self._inputs, self._outputs, self._allocations, self._batch_size = (
            allocate_bindings(
                self._engine,
                self._context,
                pagelocked_mem=pagelocked_mem,
            )
        )
        self._input_allocations: list[int] = [
            input_b.allocation for input_b in self._inputs
        ]
        self._output_allocations: list[int] = [
            output_b.allocation for output_b in self._outputs
        ]

        # store useful properties about the engine
        self._memsize: int = 0
        if FLAGS.MEMSIZE_V2:
            self._memsize = self._engine.device_memory_size_v2
        else:
            self._memsize = self._engine.device_memory_size

        # store cache random data
        self._rand_input: list[np.ndarray] | None = None

    @property
    def name(self: Self) -> str:
        """The name of the engine, as the stem of the Path."""
        return self._name

    @property
    def engine(self: Self) -> trt.ICudaEngine:
        """Access the raw TensorRT CUDA engine."""
        return self._engine

    @property
    def context(self: Self) -> trt.IExecutionContext:
        """Access the TensorRT execution context for the engine."""
        return self._context

    @property
    def logger(self: Self) -> trt.ILogger:
        """Access the TensorRT logger used for the engine."""
        return self._logger

    @property
    def stream(self: Self) -> cudart.cudaStream_t:
        """Access the underlying CUDA stream."""
        return self._stream

    @property
    def memsize(self: Self) -> int:
        """The size of the engine in bytes."""
        return self._memsize

    @cached_property
    def input_spec(self: Self) -> list[tuple[list[int], np.dtype]]:
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.

        Returns
        -------
        list[tuple[list[int], np.dtype]]
            A list with two items per element, the shape and (numpy) datatype of each input tensor.

        """
        return [(i.shape, i.dtype) for i in self._inputs]

    @cached_property
    def input_shapes(self: Self) -> list[tuple[int, ...]]:
        """
        Get the shapes for the input tensors of the network.

        Returns
        -------
        list[tuple[int, ...]]
            A list with the shape of each input tensor.

        """
        return [tuple(i.shape) for i in self._inputs]

    @cached_property
    def input_dtypes(self: Self) -> list[np.dtype]:
        """
        Get the datatypes for the input tensors of the network.

        Returns
        -------
        list[np.dtype]
            A list with the datatype of each input tensor.

        """
        return [i.dtype for i in self._inputs]

    @cached_property
    def output_spec(self: Self) -> list[tuple[list[int], np.dtype]]:
        """
        Get the specs for the output tensor of the network. Useful to prepare memory allocations.

        Returns
        -------
        list[tuple[list[int], np.dtype]]
            A list with two items per element, the shape and (numpy) datatype of each output tensor.

        """
        return [(o.shape, o.dtype) for o in self._outputs]

    @cached_property
    def output_shapes(self: Self) -> list[tuple[int, ...]]:
        """
        Get the shapes for the output tensors of the network.

        Returns
        -------
        list[tuple[int, ...]]
            A list with the shape of each output tensor.

        """
        return [tuple(o.shape) for o in self._outputs]

    @cached_property
    def output_dtypes(self: Self) -> list[np.dtype]:
        """
        Get the datatypes for the output tensors of the network.

        Returns
        -------
        list[np.dtype]
            A list with the datatype of each output tensor.

        """
        return [o.dtype for o in self._outputs]

    def __del__(self: Self) -> None:
        # Ensure CUDA stream is synchronized before freeing resources
        # This prevents issues in multithreaded environments
        with contextlib.suppress(Exception):
            stream_synchronize(self._stream)

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
                self._rng.random(size=shape, dtype=np.float32).astype(dtype)
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
