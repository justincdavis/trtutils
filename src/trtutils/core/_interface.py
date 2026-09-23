# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
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
import nvtx

from trtutils._flags import FLAGS
from trtutils._log import LOG

from ._bindings import Binding, allocate_bindings
from ._buffer import Buffer
from ._device import Device
from ._engine import create_engine, get_engine_names

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.random import Generator
    from typing_extensions import Self

    from trtutils.compat._libs import cuda, cudart, trt


class TRTEngineInterface(ABC):
    def __init__(
        self: Self,
        engine_path: Path | str,
        stream: cuda.cudaStream_t | None = None,
        dla_core: int | None = None,
        device: int | None = None,
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
        device : int, optional
            The CUDA device index to use for this engine. Default is None,
            which uses the current device.
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

        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(f"engine_interface::init [{self.name}]")
        self._dla_core = dla_core
        self._device = device
        self._device_guard = Device(device)
        self._pagelocked_mem = pagelocked_mem if pagelocked_mem is not None else True
        self._unified_mem = unified_mem if unified_mem is not None else FLAGS.IS_JETSON
        self._verbose = verbose

        with self._device_guard:
            # engine, context, logger, and CUDA stream
            self._engine, self._context, self._logger, self._stream = create_engine(
                engine_path,
                stream=stream,
                dla_core=dla_core,
                device=device,
                no_warn=no_warn,
            )

            # get the input and output names
            self._input_names, self._output_names = get_engine_names(self._engine)

            # allocate memory for inputs and outputs
            self._inputs, self._outputs, self._allocations = allocate_bindings(
                self._engine,
                self._context,
                pagelocked_mem=self._pagelocked_mem,
                unified_mem=self._unified_mem,
            )
            # engine-level input shapes (-1 marks a dynamic dim) and the
            # per-dim bounds an input Buffer's shape must fall inside
            self._input_engine_shapes: list[tuple[int, ...]] = []
            self._input_min_shapes: list[tuple[int, ...]] = []
            self._input_max_shapes: list[tuple[int, ...]] = []
            for i_binding in self._inputs:
                engine_shape, min_shape, max_shape = self._input_shape_bounds(i_binding)
                self._input_engine_shapes.append(engine_shape)
                self._input_min_shapes.append(min_shape)
                self._input_max_shapes.append(max_shape)

        # store useful properties about the engine
        self._memsize: int = 0
        if FLAGS.MEMSIZE_V2:
            self._memsize = self._engine.device_memory_size_v2
        else:
            self._memsize = self._engine.device_memory_size

        # additional verbose output about loaded engine information
        if self._verbose:
            LOG.info(f"Loaded engine: {self._name}")
            LOG.info(f"\tDevice: {self._device}")
            LOG.info(f"\tDLA Core: {self._dla_core}")
            LOG.info(f"\tPagelocked Mem: {self._pagelocked_mem}")
            LOG.info(f"\tUnified Mem: {self._unified_mem}")
            LOG.info(f"\tMemsize: {self._memsize}")
            for i_binding in self._inputs:
                LOG.info(f"\tInput: {i_binding.name} {i_binding.shape} {i_binding.dtype}")
            for o_binding in self._outputs:
                LOG.info(f"\tOutput: {o_binding.name} {o_binding.shape} {o_binding.dtype}")

        # store cache random data
        self._rand_input: list[Buffer] | None = None

        # setup the nvtx tags
        self._nvtx_tags: dict[str, str] = {
            "warmup": f"engine_interface::warmup [{self.name}]",
            "mock_execute": f"engine_interface::mock_execute [{self.name}]",
        }

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # init

    def _input_shape_bounds(
        self: Self,
        binding: Binding,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """Get the engine shape and the (min, max) profile shapes of an input."""
        if FLAGS.TRT_10:
            engine_shape = tuple(self._engine.get_tensor_shape(binding.name))
        else:
            engine_shape = tuple(self._engine.get_binding_shape(binding.index))
        if all(dim >= 0 for dim in engine_shape):
            return engine_shape, engine_shape, engine_shape
        if FLAGS.TRT_10:
            profile = self._engine.get_tensor_profile_shape(binding.name, 0)
        else:
            profile = self._engine.get_profile_shape(0, binding.name)
        return engine_shape, tuple(profile[0]), tuple(profile[2])

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
    def device(self: Self) -> int | None:
        """The CUDA device assigned to the engine."""
        return self._device

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
    def batch_size(self: Self) -> int:
        """
        Get the batch size of the engine (first dim of first input).

        For a dynamic-batch engine this is the max profile batch size.

        Returns
        -------
        int
            The batch size. Returns 1 if there are no inputs.

        """
        if self._inputs and len(self._inputs[0].shape) > 0:
            return self._inputs[0].shape[0]
        return 1

    @cached_property
    def is_dynamic_batch(self: Self) -> bool:
        """
        Check if the engine has a dynamic batch size (-1 in the first engine dim).

        Returns
        -------
        bool
            True if the engine has dynamic batch size.

        """
        if self._input_engine_shapes and len(self._input_engine_shapes[0]) > 0:
            return self._input_engine_shapes[0][0] == -1
        return False

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

    @property
    def input_names(self: Self) -> list[str]:
        """
        Get the names of the input tensors of the network.

        Returns
        -------
        list[str]
            A list with the name of each input tensor.

        """
        return self._input_names

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
    def output_names(self: Self) -> list[str]:
        """
        Get the names of the output tensors of the network.

        Returns
        -------
        list[str]
            A list with the name of each output tensor.

        """
        return self._output_names

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
        data: Sequence[Buffer],
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
        debug: bool | None = None,
    ) -> list[np.ndarray]:
        """
        Execute the network with the given inputs.

        Parameters
        ----------
        data : Sequence[Buffer]
            One Buffer per engine input, on the host or the device.
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
    def raw_exec(
        self: Self,
        data: Sequence[Buffer],
        *,
        verbose: bool | None = None,
        debug: bool | None = None,
    ) -> list[Buffer]:
        """
        Enqueue the network on its stream, leaving the outputs on the device.

        Parameters
        ----------
        data : Sequence[Buffer]
            One Buffer per engine input, on the host or the device.
        verbose : bool, optional
            Whether or not to output additional information
            to stdout. If not provided, will default to overall
            engines verbose setting.
        debug : bool, optional
            Enable intermediate stream synchronize for debugging.

        Returns
        -------
        list[Buffer]
            Device views of the outputs, shaped to the executed input shapes.

        """

    @cached_property
    def _rng(self: Self) -> Generator:
        return np.random.default_rng()

    def get_random_input(
        self: Self, *, new: bool | None = None, verbose: bool | None = None
    ) -> list[Buffer]:
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
        list[Buffer]
            One random host Buffer per engine input, at the max input shape.

        """
        verbose = verbose if verbose is not None else self._verbose
        if new or self._rand_input is None:
            # generate in input datatype directly instead of casting (if possible)
            rand_input: list[Buffer] = []
            for shape, dtype in self.input_spec:
                if np.issubdtype(dtype, np.floating):
                    rand_arr = self._rng.random(size=shape, dtype=dtype)
                else:
                    # fallback to cast if not supported
                    rand_arr = self._rng.random(size=shape, dtype=np.float32).astype(dtype)
                rand_input.append(Buffer.wrap(rand_arr))
            self._rand_input = rand_input
            if verbose:
                LOG.debug(
                    f"Generated random input: {[(b.shape, b.dtype) for b in self._rand_input]}"
                )
            return self._rand_input
        if verbose:
            LOG.debug(f"Using random input: {[(b.shape, b.dtype) for b in self._rand_input]}")
        return self._rand_input

    def __call__(
        self: Self,
        data: Sequence[Buffer],
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
        debug: bool | None = None,
    ) -> list[np.ndarray]:
        """
        Execute the network with the given inputs.

        Parameters
        ----------
        data : Sequence[Buffer]
            One Buffer per engine input, on the host or the device.
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
        data: Sequence[Buffer] | None = None,
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
        data : Sequence[Buffer], optional
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

        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["mock_execute"])

        if data is None:
            data = self.get_random_input(verbose=verbose)
        output = self.execute(data, no_copy=True, verbose=verbose, debug=debug)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # mock_execute

        return output

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

        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["warmup"])

        for _ in range(iterations):
            self.mock_execute(verbose=verbose, debug=debug)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # warmup
