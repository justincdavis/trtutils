# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING

import numpy as np

from ._flags import FLAGS
from ._log import LOG
from .core._graph import CUDAGraph
from .core._interface import TRTEngineInterface
from .core._memory import (
    memcpy_device_to_host,
    memcpy_device_to_host_async,
    memcpy_host_to_device,
    memcpy_host_to_device_async,
)
from .core._stream import stream_synchronize

if TYPE_CHECKING:
    from pathlib import Path
    from typing import ClassVar

    from typing_extensions import Self

    with contextlib.suppress(Exception):
        try:
            import cuda.bindings.driver as cuda
        except (ImportError, ModuleNotFoundError):
            from cuda import cuda


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
        stream: cuda.cudaStream_t | None = None,
        dla_core: int | None = None,
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = None,
        no_warn: bool | None = None,
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
        stream : cuda.cudaStream_t, optional
            The CUDA stream to use for this engine.
            By default None, will allocate a new stream.
        dla_core : int, optional
            The DLA core to assign DLA layers of the engine to. Default is None.
            If None, any DLA layers will be assigned to DLA core 0.
        warmup_iterations : int, optional
            The number of warmup iterations to do, by default 5
        pagelocked_mem : bool, optional
            Whether or not to use pagelocked memory for host allocations.
            By default None, which means pagelocked memory will be used.
        unified_mem : bool, optional
            Whether or not the system has unified memory.
            If True, use cudaHostAllocMapped to take advantage of unified memory.
            By default None, which will automatically determine what to use.
        cuda_graph : bool, optional
            Whether to enable CUDA graph capture for optimized execution.
            By default True. Only effective when using async_v3 backend.
        no_warn : bool, optional
            If True, suppresses warnings from TensorRT during engine deserialization.
            Default is None, which means warnings will be shown.
        verbose : bool, optional
            Whether or not to give additional information over stdout.

        Raises
        ------
        ValueError
            If the backend is not valid.

        """
        super().__init__(
            engine_path,
            stream=stream,
            dla_core=dla_core,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
            no_warn=no_warn,
            verbose=verbose,
        )

        # solve for execution method
        # only care about v2 or v3 async
        if backend not in TRTEngine._backends:
            err_msg = f"Invalid backend {backend}, options are: {TRTEngine._backends}"
            raise ValueError(err_msg)

        self._async_v3 = FLAGS.EXEC_ASYNC_V3 and (backend == "async_v3" or backend == "auto")

        # CUDA graph support
        # needs to happen before input/output bindings are set since
        # CUDA graph is used in those calls
        self._cuda_graph_enabled: bool = (
            cuda_graph if cuda_graph is not None else True
        ) and self._async_v3
        self._cuda_graph: CUDAGraph | None = None
        if self._cuda_graph_enabled:
            self._cuda_graph = CUDAGraph(self._stream)

        # if using v3
        # 1.) need to do set_input_shape for all input bindings
        # 2.) need to do set_tensor_address for all input/output bindings
        if self._async_v3:
            self._set_input_bindings()
            self._set_output_bindings()
        # if using the v3 backend also need to track if we are pointing to the 'built-in' tensors
        # only applies to the inputs
        self._using_engine_tensors: bool = True

        # store verbose info
        self._verbose = verbose if verbose is not None else False

        # store timing variable for sleep call before stream_sync
        self._sync_t: float = 0.0

        self._warmup = warmup
        if self._warmup:
            self.warmup(warmup_iterations, verbose=self._verbose)

        LOG.debug(f"Creating TRTEngine: {self.name}")

    def _set_input_bindings(self: Self) -> None:
        for i_binding in self._inputs:
            self._context.set_input_shape(i_binding.name, i_binding.shape)
            self._context.set_tensor_address(i_binding.name, i_binding.allocation)
        # CUDA graph is invalid if using new bindings
        if self._cuda_graph and self._cuda_graph.is_captured:
            self._cuda_graph.invalidate()

    def _set_output_bindings(self: Self) -> None:
        for o_binding in self._outputs:
            self._context.set_tensor_address(o_binding.name, o_binding.allocation)
        # CUDA graph is invalid if using new bindings
        if self._cuda_graph and self._cuda_graph.is_captured:
            self._cuda_graph.invalidate()

    def _capture_cuda_graph(self: Self) -> None:
        if self._cuda_graph is None:
            err_msg = f"CUDA graph is not enabled in engine: {self._name}"
            raise RuntimeError(err_msg)

        # at least one execution required prior to graph capture
        # simply use one warmup iteration if warmup didnt get run
        if not self._warmup:
            self.warmup(1, verbose=self._verbose)

        # CUDAGraph handles capture with a context manager
        with self._cuda_graph:
            # manually run execute_async_v3 instead of execute since
            # we only want the TRT engine
            self._context.execute_async_v3(self._stream)

    def __del__(self: Self) -> None:
        with contextlib.suppress(AttributeError):
            if self._cuda_graph is not None:
                self._cuda_graph.invalidate()
        super().__del__()

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
        verbose = verbose if verbose is not None else self._verbose
        if verbose:
            LOG.info(f"{time.perf_counter()} {self.name} Dispatch: BEGIN")

        # reset the input bindings if direct_exec or raw_exec were used
        if not self._using_engine_tensors:
            self._set_input_bindings()
            self._using_engine_tensors = True

        # copy inputs
        if self._pagelocked_mem and self._unified_mem:
            for i_idx in range(len(self._inputs)):
                np.copyto(self._inputs[i_idx].host_allocation, data[i_idx])
        elif self._pagelocked_mem:
            for i_idx in range(len(self._inputs)):
                memcpy_host_to_device_async(
                    self._inputs[i_idx].allocation,
                    data[i_idx],
                    self._stream,
                )
        else:
            for i_idx in range(len(self._inputs)):
                memcpy_host_to_device(
                    self._inputs[i_idx].allocation,
                    data[i_idx],
                )

        if debug:
            stream_synchronize(self._stream)

        # execute
        if self._cuda_graph:
            if self._cuda_graph.is_captured:
                # uses already captured graph to handle execution
                self._cuda_graph.launch()
            else:
                # CUDA graph capture calls execute_async_v3 internally
                # no need to call again here
                self._capture_cuda_graph()
        # base execution cases
        elif self._async_v3:
            self._context.execute_async_v3(self._stream)
        else:
            self._context.execute_async_v2(self._allocations, self._stream)

        if debug:
            stream_synchronize(self._stream)

        # copy outputs
        if self._unified_mem and self._pagelocked_mem:
            pass
        elif self._pagelocked_mem:
            for o_idx in range(len(self._outputs)):
                memcpy_device_to_host_async(
                    self._outputs[o_idx].host_allocation,
                    self._outputs[o_idx].allocation,
                    self._stream,
                )
        else:
            for o_idx in range(len(self._outputs)):
                memcpy_device_to_host(
                    self._outputs[o_idx].host_allocation,
                    self._outputs[o_idx].allocation,
                )

        # make sure all operations are complete
        stream_synchronize(self._stream)

        if verbose:
            LOG.info(f"{time.perf_counter()} {self.name} Dispatch: END")

        # return the results
        if no_copy:
            return [o.host_allocation for o in self._outputs]
        return [o.host_allocation.copy() for o in self._outputs]

    def direct_exec(
        self: Self,
        pointers: list[int],
        *,
        set_pointers: bool = True,
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
            Pointers must be in the order of expected inputs for the engine.
        set_pointers : bool, optional
            Whether to set tensor addresses before execution.
            If True (default), tensor addresses will be set.
            If False, tensor addresses are assumed to already be configured.
            By default True.
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
        verbose = verbose if verbose is not None else self._verbose
        if not no_warn:
            LOG.warning(
                "Calling direct_exec is potentially dangerous, ensure all pointers and data are valid. Outputs can be overwritten inplace!",
            )

        # execute
        if self._async_v3:
            if set_pointers:
                # need to set the input pointers to match the bindings, assume in same order
                for i in range(len(pointers)):
                    self._context.set_tensor_address(self._inputs[i].name, pointers[i])
                self._using_engine_tensors = (
                    False  # set flag to tell future execute calls to reset inputs
                )
            self._context.execute_async_v3(self._stream)
        else:
            self._context.execute_async_v2(
                pointers + self._output_allocations,
                self._stream,
            )

        if debug:
            stream_synchronize(self._stream)

        # copy outputs
        if self._unified_mem and self._pagelocked_mem:
            pass
        elif self._pagelocked_mem:
            for o_idx in range(len(self._outputs)):
                memcpy_device_to_host_async(
                    self._outputs[o_idx].host_allocation,
                    self._outputs[o_idx].allocation,
                    self._stream,
                )
        else:
            for o_idx in range(len(self._outputs)):
                memcpy_device_to_host(
                    self._outputs[o_idx].host_allocation,
                    self._outputs[o_idx].allocation,
                )

        # make sure all operations are complete
        stream_synchronize(self._stream)

        # return the output host allocations
        return self._output_host_allocations

    def raw_exec(
        self: Self,
        pointers: list[int],
        *,
        set_pointers: bool = True,
        no_warn: bool | None = None,
        verbose: bool | None = None,
        debug: bool | None = None,
    ) -> list[int]:
        """
        Execute the network with the given GPU memory pointers.

        The outputs of this function are the direct GPU pointers
        of the output allocations.

        Parameters
        ----------
        pointers : list[int]
            The inputs to the network.
            Pointers must be in the order of expected inputs for the engine.
        set_pointers : bool, optional
            Whether to set tensor addresses before execution.
            If True (default), tensor addresses will be set.
            If False, tensor addresses are assumed to already be configured.
            By default True.
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
        list[int]
            The pointers to the network outputs.

        """
        verbose = verbose if verbose is not None else self._verbose
        if not no_warn:
            LOG.warning(
                "Calling raw_exec is potentially dangerous, ensure all pointers and data are valid. Outputs can be overwritten inplace!",
            )

        # execute
        if self._async_v3:
            if set_pointers:
                # need to set the input pointers to match the bindings, assume in same order
                for i in range(len(pointers)):
                    self._context.set_tensor_address(self._inputs[i].name, pointers[i])
                self._using_engine_tensors = (
                    False  # set flag to tell future execute calls to reset inputs
                )
            self._context.execute_async_v3(self._stream)
        else:
            self._context.execute_async_v2(
                pointers + self._output_allocations,
                self._stream,
            )

        if debug:
            stream_synchronize(self._stream)

        # return the pointers to the output allocations
        return self._output_allocations
