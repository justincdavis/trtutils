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
from trtutils._log import LOG

from ._bindings import Binding, allocate_bindings
from ._engine import create_engine

if TYPE_CHECKING:
    from typing_extensions import Self

    with contextlib.suppress(ImportError):
        import tensorrt as trt

        try:
            import cuda.bindings.driver as cuda
            import cuda.bindings.runtime as cudart
        except (ImportError, ModuleNotFoundError):
            from cuda import cuda, cudart


class TRTEngineInterface(ABC):
    def __init__(
        self: Self,
        engine_path: Path | str,
        stream: cuda.cudaStream_t | None = None,
        dla_core: int | None = None,
        *,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        """
        Load the TensorRT engine from a file.

        Parameters
        ----------
        engine_path : Path | str
            The path to the serialized engine file.
        stream : cuda.cudaStream_t, optional
            The CUDA stream to use for this engine.
            By default None, will allocate a new stream.
        dla_core : int, optional
            The DLA core to assign DLA layers of the engine to. Default is None.
            If None, any DLA layers will be assigned to DLA core 0.
        pagelocked_mem : bool, optional
            Whether or not to use pagelocked memory for host allocations.
            By default None, which means pagelocked memory will be used.
        unified_mem : bool, optional
            Whether or not the system has unified memory.
            If True, use cudaHostAllocMapped to take advantage of unified memory.
            By default None, which means the default host allocation will be used.
        no_warn : bool, optional
            If True, suppresses warnings from TensorRT during engine deserialization.
            Default is None, which means warnings will be shown.
        verbose : bool, optional
            Whether or not to give additional information over stdout.

        """
        # store path stem as name
        self._name = Path(engine_path).stem
        self._dla_core = dla_core
        self._pagelocked_mem = pagelocked_mem if pagelocked_mem is not None else True
        self._unified_mem = unified_mem if unified_mem is not None else FLAGS.IS_JETSON
        self._verbose = verbose

        # engine, context, logger, and CUDA stream
        self._engine, self._context, self._logger, self._stream = create_engine(
            engine_path,
            stream=stream,
            dla_core=dla_core,
            no_warn=no_warn,
        )

        # allocate memory for inputs and outputs
        self._inputs, self._outputs, self._allocations = allocate_bindings(
            self._engine,
            self._context,
            pagelocked_mem=self._pagelocked_mem,
            unified_mem=self._unified_mem,
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

    @property
    def dla_core(self: Self) -> int | None:
        """The DLA core assigned to the engine."""
        return self._dla_core

    @property
    def pagelocked_mem(self: Self) -> bool:
        """Whether or not the system has pagelocked memory."""
        return self._pagelocked_mem

    @property
    def unified_mem(self: Self) -> bool:
        """Whether or not the system has unified memory."""
        return self._unified_mem

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

    @property
    def input_bindings(self: Self) -> list[Binding]:
        """
        Get the input bindings.

        Returns
        -------
        list[Binding]
            The input bindings.

        """
        return self._inputs

    @property
    def output_bindings(self: Self) -> list[Binding]:
        """
        Get the output bindings.

        Returns
        -------
        list[Binding]
            The output bindings.

        """
        return self._outputs

    def __del__(self: Self) -> None:
        # NOTE: handle stream sync/cleanup better
        # # Ensure CUDA stream is synchronized before freeing resources
        # # This prevents issues in multithreaded environments
        # with contextlib.suppress(Exception):
        #     stream_synchronize(self._stream)

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
        verbose: bool | None = None,
        debug: bool | None = None,
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
        verbose : bool, optional
            Whether or not to output additional information
            to stdout. If not provided, will default to overall
            engines verbose setting.
        debug : bool, optional
            Enable intermediate stream synchronize for debugging.

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
        verbose: bool | None = None,
        debug: bool | None = None,
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
        verbose : bool, optional
            Whether or not to output additional information
            to stdout. If not provided, will default to overall
            engines verbose setting.
        debug : bool, optional
            Enable intermediate stream synchronize for debugging.

        Returns
        -------
        list[np.ndarray]
            The outputs of the network.

        """

    @cached_property
    def _rng(self: Self) -> np.random.Generator:
        return np.random.default_rng()

    def get_random_input(
        self: Self, *, new: bool | None = None, verbose: bool | None = None
    ) -> list[np.ndarray]:
        """
        Generate a random input for the network.

        Parameters
        ----------
        new : bool, optional
            Whether or not to generate new input. By default None/False.
        verbose : bool, optional
            Whether or not to output additional information
            to stdout. If not provided, will default to overall
            engines verbose setting.

        Returns
        -------
        list[np.ndarray]
            The random input to the network.

        """
        verbose = verbose if verbose is not None else self._verbose
        if new or self._rand_input is None:
            # generate in input datatype directly instead of casting (if possible)
            rand_input = []
            for shape, dtype in self.input_spec:
                if np.issubdtype(dtype, np.floating):
                    rand_arr = self._rng.random(size=shape, dtype=dtype)
                else:
                    # fallback to cast if not supported
                    rand_arr = self._rng.random(size=shape, dtype=np.float32).astype(
                        dtype
                    )
                rand_input.append(rand_arr)
            self._rand_input = rand_input
            if verbose:
                LOG.debug(
                    f"Generated random input: {[(a.shape, a.dtype) for a in self._rand_input]}"
                )
            return self._rand_input
        if verbose:
            LOG.debug(
                f"Using random input: {[(a.shape, a.dtype) for a in self._rand_input]}"
            )
        return self._rand_input

    def __call__(
        self: Self,
        data: list[np.ndarray],
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
        debug: bool | None = None,
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
        verbose : bool, optional
            Whether or not to output additional information
            to stdout. If not provided, will default to overall
            engines verbose setting.
        debug : bool, optional
            Enable intermediate stream synchronize for debugging.

        Returns
        -------
        list[np.ndarray]
            The outputs of the network.

        """
        return self.execute(data, no_copy=no_copy, verbose=verbose, debug=debug)

    def mock_execute(
        self: Self,
        data: list[np.ndarray] | None = None,
        *,
        verbose: bool | None = None,
        debug: bool | None = None,
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
        verbose : bool, optional
            Whether or not to output additional information
            to stdout. If not provided, will default to overall
            engines verbose setting.
        debug : bool, optional
            Enable intermediate stream synchronize for debugging.

        Returns
        -------
        list[np.ndarray]
            The outputs of the network.

        """
        verbose = verbose if verbose is not None else self._verbose
        if verbose:
            LOG.debug(f"Mock-execute: data={bool(data)}")
        if data is None:
            data = self.get_random_input(verbose=verbose)
        return self.execute(data, no_copy=True, verbose=verbose, debug=debug)

    def warmup(
        self: Self,
        iterations: int,
        *,
        verbose: bool | None = None,
        debug: bool | None = None,
    ) -> None:
        """
        Warmup the network for a given number of iterations.

        Parameters
        ----------
        iterations : int
            The number of iterations to warmup the network.
        verbose : bool, optional
            Whether or not to output additional information
            to stdout. If not provided, will default to overall
            engines verbose setting.
        debug : bool, optional
            Enable intermediate stream synchronize for debugging.

        """
        verbose = verbose if verbose is not None else self._verbose
        for _ in range(iterations):
            self.mock_execute(verbose=verbose, debug=debug)
