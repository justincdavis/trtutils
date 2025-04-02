# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import contextlib
import time
from queue import Empty, Queue
from threading import Thread
from typing import TYPE_CHECKING

from ._flags import FLAGS
from ._log import LOG
from .core import (
    TRTEngineInterface,
    memcpy_device_to_host_async,
    memcpy_host_to_device_async,
    stream_synchronize,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from typing import ClassVar

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

    _backends: ClassVar[set[str]] = {"auto", "async_v3", "async_v2"}

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 5,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        verbose: bool | None = None,
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
        backend : str, optional
            What version of backend execution to use.
            By default 'auto', which will use v3 if available otherwise v2.
            Options are: ['auto', 'async_v3', 'async_v2]
        warmup_iterations : int, optional
            The number of warmup iterations to do, by default 5
        pagelocked_mem : bool, optional
            Whether or not to use pagelocked memory for host allocations.
            By default None, which means pagelocked memory will be used.
        verbose : bool, optional
            Whether or not to give additional information over stdout.

        Raises
        ------
        ValueError
            If the backend is not valid.

        """
        super().__init__(engine_path, pagelocked_mem=pagelocked_mem)

        # solve for execution method
        # only care about v2 or v3 async
        if backend not in TRTEngine._backends:
            err_msg = f"Invalid backend {backend}, options are: {TRTEngine._backends}"
            raise ValueError(err_msg)

        self._async_v3 = FLAGS.EXEC_ASYNC_V3 and (
            backend == "async_v3" or backend == "auto"
        )
        # if using v3
        # 1.) need to do set_input_shape for all input bindings
        # 2.) need to do set_tensor_address for all input/output bindings
        if self._async_v3:
            for i_binding in self._inputs:
                self._context.set_input_shape(i_binding.name, i_binding.shape)
                self._context.set_tensor_address(i_binding.name, i_binding.allocation)
            for o_binding in self._outputs:
                self._context.set_tensor_address(o_binding.name, o_binding.allocation)

        # store verbose info
        self._verbose = verbose if verbose is not None else False

        # store timing variable for sleep call before stream_sync
        self._sync_t: float = 0.0

        if warmup:
            self.warmup(warmup_iterations)

        LOG.debug(f"Creating TRTEngine: {self.name}")

    def __del__(self: Self) -> None:
        super().__del__()

    def execute(
        self: Self,
        data: list[np.ndarray],
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
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

        Returns
        -------
        list[np.ndarray]
            The outputs of the network.

        """
        verbose = verbose if verbose is not None else self._verbose
        # Copy inputs
        if verbose:
            LOG.info(f"{time.perf_counter()} {self.name} Dispatch: BEGIN")
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
        if self._async_v3:
            self._context.execute_async_v3(self._stream)
        else:
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
        if verbose:
            LOG.info(f"{time.perf_counter()} {self.name} Dispatch: END")

        # # add additional sleep here to help parallel engines
        # t0 = time.time()
        # time.sleep(max(self._sync_t - 0.001, 0.0))
        # stream_synchronize(self._stream)
        # t1 = time.time()
        # self._sync_t = t1 - t0
        stream_synchronize(self._stream)

        # return
        # copy the buffer since future inference will overwrite
        if no_copy:
            return [o.host_allocation for o in self._outputs]
        return [o.host_allocation.copy() for o in self._outputs]

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
        if not no_warn:
            LOG.warning(
                "Calling direct_exec is potentially dangerous, ensure all pointers and data are valid. Outputs can be overwritten inplace!",
            )
        # execute
        if self._async_v3:
            # need to set the input pointers to match the bindings, assume in same order
            for i in range(len(pointers)):
                self._context.set_tensor_address(self._inputs[i].name, pointers[i])
            self._context.execute_async_v3(self._stream)
        else:
            self._context.execute_async_v2(
                pointers + self._output_allocations,
                self._stream,
            )
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
        return [o.host_allocation for o in self._outputs]


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
        self._engine: TRTEngine = TRTEngine(
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            warmup=warmup,
        )
        self._input_queue: Queue[list[np.ndarray]] = Queue()
        self._output_queue: Queue[list[np.ndarray]] = Queue()
        self._thread = Thread(
            target=self._run,
            args=(),
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

        """
        return self._engine.input_spec

    @property
    def input_shapes(self: Self) -> list[tuple[int, ...]]:
        """
        Get the shapes for the input tensors of the network.

        Returns
        -------
        list[tuple[int, ...]]
            A list with the shape of each input tensor.

        """
        return self._engine.input_shapes

    @property
    def input_dtypes(self: Self) -> list[np.dtype]:
        """
        Get the datatypes for the input tensors of the network.

        Returns
        -------
        list[np.dtype]
            A list with the datatype of each input tensor.

        """
        return self._engine.input_dtypes

    @property
    def output_spec(self: Self) -> list[tuple[list[int], np.dtype]]:
        """
        Get the specs for the output tensor of the network. Useful to prepare memory allocations.

        Returns
        -------
        list[tuple[list[int], np.dtype]]
            A list with two items per element, the shape and (numpy) datatype of each output tensor.

        """
        return self._engine.output_spec

    @property
    def output_shapes(self: Self) -> list[tuple[int, ...]]:
        """
        Get the shapes for the output tensors of the network.

        Returns
        -------
        list[tuple[int, ...]]
            A list with the shape of each output tensor.

        """
        return self._engine.output_shapes

    @property
    def output_dtypes(self: Self) -> list[np.dtype]:
        """
        Get the datatypes for the output tensors of the network.

        Returns
        -------
        list[np.dtype]
            A list with the datatype of each output tensor.

        """
        return self._engine.output_dtypes

    def get_random_input(self: Self, *, new: bool | None = None) -> list[np.ndarray]:
        """
        Get a random input to the underlying TRTEngine.

        Parameters
        ----------
        new : bool, optional
            Whether or not to get a new input or the cached already generated one.
            By default, None/False

        Returns
        -------
        list[np.ndarray]
            The random input.

        """
        return self._engine.get_random_input(new=new)

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
        """Send a random input to the engine."""
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
    ) -> None:
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

    def get_random_input(
        self: Self,
        *,
        new: bool | None = None,
    ) -> list[list[np.ndarray]]:
        """
        Get a random input to the underlying TRTEngines.

        Parameters
        ----------
        new : bool, optional
            Whether or not to get a new input or the cached already generated one.
            By default, None/False

        Returns
        -------
        list[list[np.ndarray]]
            The random inputs.

        """
        return [e.get_random_input(new=new) for e in self._engines]

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
