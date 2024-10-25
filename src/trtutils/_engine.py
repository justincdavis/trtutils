# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import contextlib
from functools import cached_property
from queue import Empty, Queue
from threading import Thread
from typing import TYPE_CHECKING

from .core import (
    TRTEngineInterface,
    allocate_bindings,
    memcpy_device_to_host_async,
    memcpy_host_to_device_async,
    stream_synchronize,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import numpy as np
    from typing_extensions import Self


class TRTEngine(TRTEngineInterface):
    """
    Implements a generic interface for TensorRT engines.

    It is thread and process safe to create multiple TRTEngines.
    It is valid to create a TRTEngine in one thread and use in another.
    Each TRTEngine has its own CUDA context and there is no safeguards
    implemented in the class for datarace conditions. As such, a
    single TRTEngine should not be used in multiple threads or processes.
    """

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
        super().__init__(engine_path)

        # allocate memory for inputs and outputs
        self._inputs, self._outputs, self._allocations, self._batch_size = (
            allocate_bindings(
                self._engine,
                self._context,
                self._logger,
            )
        )

        if warmup:
            for _ in range(warmup_iterations):
                self.mock_execute()

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

    def execute(self: Self, data: list[np.ndarray]) -> list[np.ndarray]:
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
        # Copy inputs
        for i_idx in range(len(self._inputs)):
            # memcpy_host_to_device(
            #     self._inputs[i_idx].allocation,
            #     data[i_idx],
            # )
            memcpy_host_to_device_async(
                self._inputs[i_idx].allocation,
                data[i_idx],
                self._stream,
            )
        # execute
        self._context.execute_async_v2(self._allocations, self._stream)
        # Copy outputs
        for o_idx in range(len(self._outputs)):
            # memcpy_device_to_host(
            #     self._outputs[o_idx].host_allocation,
            #     self._outputs[o_idx].allocation,
            # )
            memcpy_device_to_host_async(
                self._outputs[o_idx].host_allocation,
                self._outputs[o_idx].allocation,
                self._stream,
            )
        # sync the stream
        stream_synchronize(self._stream)
        # return
        # copy the buffer since future inference will overwrite
        return [o.host_allocation.copy() for o in self._outputs]


class QueuedTRTEngine:
    """Interact with TRTEngine over Thread and Queue."""

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 5,
        *,
        warmup: bool | None = None,
    ) -> None:
        """
        Create a QueuedTRTEngine.

        Parameters
        ----------
        engine_path : Path, str
            The Path to the compiled TensorRT engine.
        warmup_iterations : int
            The number of iterations to warmup the engine.
            By default 5
        warmup : bool, optional
            Whether or not to perform warmup iterations.

        """
        self._stopped = False  # flag for if user stopped thread
        self._engine: TRTEngine | None = None  # storage for engine data
        self._input_queue: Queue[list[np.ndarray]] = Queue()
        self._output_queue: Queue[list[np.ndarray]] = Queue()
        self._thread = Thread(
            target=self._run,
            kwargs={
                "engine_path": engine_path,
                "warmup_iterations": warmup_iterations,
                "warmup": warmup,
            },
            daemon=True,
        )
        self._thread.start()

    def __del__(self: Self) -> None:
        self.stop()

    @property
    def input_spec(self: Self) -> list[tuple[list[int], np.dtype]]:
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.

        Returns
        -------
        list[tuple[list[int], np.dtype]]
            A list with two items per element, the shape and (numpy) datatype of each input tensor.

        Raises
        ------
        RuntimeError
            The engine has not been created yet.

        """
        if self._engine is None:
            err_msg = "Engine has not been created yet."
            raise RuntimeError(err_msg)
        return self._engine.input_spec

    @property
    def input_shapes(self: Self) -> list[tuple[int, ...]]:
        """
        Get the shapes for the input tensors of the network.

        Returns
        -------
        list[tuple[int, ...]]
            A list with the shape of each input tensor.

        Raises
        ------
        RuntimeError
            The engine has not been created yet.

        """
        if self._engine is None:
            err_msg = "Engine has not been created yet."
            raise RuntimeError(err_msg)
        return self._engine.input_shapes

    @property
    def input_dtypes(self: Self) -> list[np.dtype]:
        """
        Get the datatypes for the input tensors of the network.

        Returns
        -------
        list[np.dtype]
            A list with the datatype of each input tensor.

        Raises
        ------
        RuntimeError
            The engine has not been created yet.

        """
        if self._engine is None:
            err_msg = "Engine has not been created yet."
            raise RuntimeError(err_msg)
        return self._engine.input_dtypes

    @property
    def output_spec(self: Self) -> list[tuple[list[int], np.dtype]]:
        """
        Get the specs for the output tensor of the network. Useful to prepare memory allocations.

        Returns
        -------
        list[tuple[list[int], np.dtype]]
            A list with two items per element, the shape and (numpy) datatype of each output tensor.

        Raises
        ------
        RuntimeError
            The engine has not been created yet.

        """
        if self._engine is None:
            err_msg = "Engine has not been created yet."
            raise RuntimeError(err_msg)
        return self._engine.output_spec

    @property
    def output_shapes(self: Self) -> list[tuple[int, ...]]:
        """
        Get the shapes for the output tensors of the network.

        Returns
        -------
        list[tuple[int, ...]]
            A list with the shape of each output tensor.

        Raises
        ------
        RuntimeError
            The engine has not been created yet.

        """
        if self._engine is None:
            err_msg = "Engine has not been created yet."
            raise RuntimeError(err_msg)
        return self._engine.output_shapes

    @property
    def output_dtypes(self: Self) -> list[np.dtype]:
        """
        Get the datatypes for the output tensors of the network.

        Returns
        -------
        list[np.dtype]
            A list with the datatype of each output tensor.

        Raises
        ------
        RuntimeError
            The engine has not been created yet.

        """
        if self._engine is None:
            err_msg = "Engine has not been created yet."
            raise RuntimeError(err_msg)
        return self._engine.output_dtypes

    def stop(
        self: Self,
    ) -> None:
        """Stop the thread containing the TRTEngine."""
        self._stopped = True
        self._thread.join()

    def submit(
        self: Self,
        data: list[np.ndarray],
    ) -> None:
        """
        Put data in the input queue.

        Parameters
        ----------
        data : list[np.ndarray]
            The data to have the engine run.

        """
        self._input_queue.put(data)

    def mock_submit(
        self: Self,
    ) -> None:
        """
        Send a random input to the engine.

        Raises
        ------
        RuntimeError
            If the engine has not been created

        """
        if self._engine is None:
            err_msg = "Engine has not been created."
            raise RuntimeError(err_msg)
        data = self._engine.get_random_input()
        self._input_queue.put(data)

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
        engine_path: Path,
        warmup_iterations: int,
        *,
        warmup: bool,
    ) -> None:
        self._engine = TRTEngine(
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            warmup=warmup,
        )

        while not self._stopped:
            try:
                inputs = self._input_queue.get(timeout=0.1)
            except Empty:
                continue

            result = self._engine(inputs)

            self._output_queue.put(result)


class ParallelTRTEngines:
    """Handle many TRTEngines in parallel."""

    def __init__(
        self: Self,
        engine_paths: Sequence[Path | str],
        warmup_iterations: int = 5,
        *,
        warmup: bool | None = None,
    ) -> None:
        """
        Create a ParallelTRTEngines instance.

        Parameters
        ----------
        engine_paths : Sequence[Path | str]
            The Paths to the compiled engines to use.
        warmup_iterations : int
            The number of iteratiosn to perform warmup for.
            By default 5
        warmup : bool, optional
            Whether or not to run warmup iterations on the engines.

        """
        self._engines: list[QueuedTRTEngine] = [
            QueuedTRTEngine(
                engine_path=epath,
                warmup_iterations=warmup_iterations,
                warmup=warmup,
            )
            for epath in engine_paths
        ]

    def stop(self: Self) -> None:
        """Stop the underlying engine threads."""
        for engine in self._engines:
            engine.stop()

    def submit(
        self: Self,
        inputs: list[list[np.ndarray]],
    ) -> None:
        """
        Submit data to be processed by the engines.

        Parameters
        ----------
        inputs : list[list[np.ndarray]]
            The inputs to pass to the engines.
            Should be a list of the same lenght of engines created.

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
            engine.submit(data)

    def mock_submit(
        self: Self,
    ) -> None:
        """Send random data to the engines."""
        for engine in self._engines:
            engine.mock_submit()

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
