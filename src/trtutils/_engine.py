# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib
import math
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import nvtx

from ._flags import FLAGS
from ._log import LOG
from .core._graph import CUDAGraph
from .core._interface import TRTEngineInterface
from .core._stream import stream_synchronize

if TYPE_CHECKING:
    from typing import ClassVar

    import numpy as np
    from typing_extensions import Self

    from trtutils.compat._libs import cuda


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
    _capture_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 5,
        backend: str = "auto",
        stream: cuda.cudaStream_t | None = None,
        dla_core: int | None = None,
        device: int | None = None,
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
        device : int, optional
            The CUDA device index to use for this engine. Default is None,
            which uses the current device.
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
        self._name = Path(engine_path).stem

        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(f"engine::init [{self.name}]")

        super().__init__(
            engine_path,
            stream=stream,
            dla_core=dla_core,
            device=device,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
            no_warn=no_warn,
            verbose=verbose,
        )

        self._nvtx_tags.update(
            {
                "graph_capture": f"engine::graph_capture [{self.name}]",
                "execute": f"engine::execute [{self.name}]",
                "graph_exec": f"engine::graph_exec [{self.name}]",
                "direct_exec": f"engine::direct_exec [{self.name}]",
                "raw_exec": f"engine::raw_exec [{self.name}]",
            }
        )

        # solve for execution method
        # only care about v2 or v3 async
        if backend not in TRTEngine._backends:
            err_msg = f"Invalid backend {backend}, options are: {TRTEngine._backends}"
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()  # init
            raise ValueError(err_msg)

        self._async_v3 = FLAGS.EXEC_ASYNC_V3 and (backend == "async_v3" or backend == "auto")

        # CUDA graph support
        # needs to happen before input/output bindings are set since
        # CUDA graph is used in those calls
        self._cuda_graph_enabled: bool = (
            cuda_graph if cuda_graph is not None else True
        ) and self._async_v3
        self._cuda_graph: CUDAGraph | None = None
        self._capturing_graph: bool = False  # Guard against capture recursion
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

        # dynamic-shape tracking (only supported on the v3 backend)
        # bindings are allocated at the max profile shape, so partial batches
        # always fit; the context shape is adjusted per-call to avoid computing
        # and copying at the max batch size when a smaller batch is submitted
        self._has_dynamic_inputs: bool = bool(self._dynamic_input_names) and self._async_v3
        self._active_input_shapes: list[tuple[int, ...]] = [tuple(b.shape) for b in self._inputs]
        # None -> context is at the allocated (max) shapes, use host allocations as-is
        self._active_output_shapes: list[tuple[int, ...]] | None = None
        # element counts matching _active_output_shapes, cached to avoid
        # math.prod per output per call in the hot path
        self._active_output_sizes: list[int] | None = None

        # store timing variable for sleep call before stream_sync
        self._sync_t: float = 0.0

        # store verbose info
        self._verbose = verbose if verbose is not None else False

        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["warmup"])

        self._warmup = warmup
        if self._warmup:
            self.warmup(warmup_iterations, verbose=self._verbose)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # warmup
            nvtx.pop_range()  # init

        LOG.debug(f"Created TRTEngine: {self.name}")

    def _set_input_bindings(self: Self) -> None:
        for i_binding in self._inputs:
            self._context.set_input_shape(i_binding.name, i_binding.shape)
            self._context.set_tensor_address(i_binding.name, i_binding.allocation)
        # context is back at the allocated (max) shapes
        self._active_input_shapes = [tuple(b.shape) for b in self._inputs]
        self._active_output_shapes = None
        self._active_output_sizes = None
        # CUDA graph is invalid if using new bindings
        if self._cuda_graph and self._cuda_graph.is_captured:
            self._cuda_graph.invalidate()

    def _set_output_bindings(self: Self) -> None:
        for o_binding in self._outputs:
            self._context.set_tensor_address(o_binding.name, o_binding.allocation)
        # CUDA graph is invalid if using new bindings
        if self._cuda_graph and self._cuda_graph.is_captured:
            self._cuda_graph.invalidate()

    def _apply_input_shapes(self: Self, data: list[np.ndarray]) -> None:
        """
        Set the context input shapes to match the submitted data.

        Only called when the engine has dynamic input dims and the v3
        backend is in use. When the submitted shapes match the currently
        active shapes this is a no-op, so static workloads that always
        submit the same shape pay a tuple comparison per input and
        nothing else.

        Parameters
        ----------
        data : list[np.ndarray]
            The inputs to the network.

        Raises
        ------
        ValueError
            If an input's shape exceeds the allocated max profile shape.

        """
        changed = False
        for i_idx, i_binding in enumerate(self._inputs):
            if i_binding.name not in self._dynamic_input_names:
                continue
            data_shape = tuple(data[i_idx].shape)
            if data_shape == self._active_input_shapes[i_idx]:
                continue
            # bindings are allocated at the max profile shape, anything
            # larger cannot fit in the allocation
            if data[i_idx].nbytes > i_binding.host_allocation.nbytes:
                err_msg = (
                    f"Input '{i_binding.name}' with shape {data_shape} exceeds the "
                    f"allocated max profile shape {tuple(i_binding.shape)} for engine "
                    f"'{self._name}'. Split the data into chunks of at most the max "
                    "profile batch size."
                )
                raise ValueError(err_msg)
            self._context.set_input_shape(i_binding.name, data_shape)
            self._active_input_shapes[i_idx] = data_shape
            changed = True
        if changed:
            # resolve the output shapes for the new input shapes so the
            # D2H copies only move the valid prefix of each output
            self._active_output_shapes = [
                tuple(self._context.get_tensor_shape(o.name)) for o in self._outputs
            ]
            # if the context is back at the allocated shapes, clear tracking
            if self._active_input_shapes == [tuple(b.shape) for b in self._inputs]:
                self._active_output_shapes = None
            self._refresh_output_sizes()

    def _refresh_output_sizes(self: Self) -> None:
        if self._active_output_shapes is None:
            self._active_output_sizes = None
        else:
            self._active_output_sizes = [math.prod(shape) for shape in self._active_output_shapes]

    def _resolve_dynamic_batch(
        self: Self,
        batch_size: int,
    ) -> list[tuple[int, ...]] | None:
        """
        Set the batch dim of every dynamic input to batch_size.

        Used by the end2end path where inputs are raw GPU pointers and
        _apply_input_shapes (which inspects host arrays) never runs.
        Shares the same shape bookkeeping as execute(), so mixing
        end2end() and execute() calls stays consistent.

        Parameters
        ----------
        batch_size : int
            The batch size to resolve the dynamic inputs to.

        Returns
        -------
        list[tuple[int, ...]] | None
            The context output shapes when the context sits at a
            non-allocated (partial) shape, otherwise None (static engine
            or batch at the max profile shape).

        Raises
        ------
        ValueError
            If batch_size exceeds the max profile batch dim.

        """
        if not self._has_dynamic_inputs:
            return None
        changed = False
        for i_idx, i_binding in enumerate(self._inputs):
            if i_binding.name not in self._dynamic_input_names:
                continue
            engine_shape = self._engine_input_shapes[i_idx]
            target = tuple(batch_size if dim < 0 else dim for dim in engine_shape)
            if target == self._active_input_shapes[i_idx]:
                continue
            max_shape = tuple(i_binding.shape)
            if any(t > m for t, m in zip(target, max_shape)):
                err_msg = (
                    f"Batch size {batch_size} exceeds the max profile shape "
                    f"{max_shape} for input '{i_binding.name}' of engine "
                    f"'{self._name}'."
                )
                raise ValueError(err_msg)
            self._context.set_input_shape(i_binding.name, target)
            self._active_input_shapes[i_idx] = target
            changed = True
        if changed:
            if self._active_input_shapes == [tuple(b.shape) for b in self._inputs]:
                self._active_output_shapes = None
            else:
                self._active_output_shapes = [
                    tuple(self._context.get_tensor_shape(o.name)) for o in self._outputs
                ]
            self._refresh_output_sizes()
        return self._active_output_shapes

    def _capture_cuda_graph(self: Self) -> None:
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["graph_capture"])

        # Prevent recursion: warmup() -> mock_execute() -> execute() -> _capture_cuda_graph()
        if self._capturing_graph:
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()  # graph_capture
            return

        if self._cuda_graph is None:
            err_msg = f"CUDA graph is not enabled in engine: {self._name}"
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()  # graph_capture
            raise RuntimeError(err_msg)

        self._capturing_graph = True
        capture_error: RuntimeError | None = None
        try:
            # serialize CUDA graph capture
            with self._capture_lock:
                # at least one execution required prior to graph capture
                # simply use one warmup iteration if warmup didnt get run
                if not self._warmup:
                    try:
                        self.warmup(1, verbose=self._verbose)
                    except RuntimeError as e:
                        # assess if cuda graph capture fails during warmup
                        if self._cuda_graph is not None:
                            self._cuda_graph.invalidate()
                        self._cuda_graph = None
                        err_msg = (
                            f"CUDA graph capture failed for engine '{self._name}' during warmup: {e}\n"
                            "This can happen when multiple engines attempt graph capture simultaneously.\n"
                            "To resolve: use cuda_graph=False, or ensure engines are created sequentially, "
                            "or use warmup=True to capture graphs at initialization time."
                        )
                        capture_error = RuntimeError(err_msg)
                        capture_error.__cause__ = e
                        return

                # capture graph
                with self._cuda_graph:
                    self._context.execute_async_v3(self._stream)

                # assess graph capture success
                if not self._cuda_graph.is_captured:
                    self._cuda_graph = None
                    err_msg = (
                        f"CUDA graph capture failed for engine '{self._name}'.\n"
                        "The engine may not support CUDA graph capture.\n"
                        "To resolve: use cuda_graph=False to disable CUDA graphs for this engine."
                    )
                    capture_error = RuntimeError(err_msg)
        finally:
            self._capturing_graph = False
            if capture_error is not None:
                if FLAGS.NVTX_ENABLED:
                    nvtx.pop_range()  # graph_capture
                raise capture_error

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # graph_capture

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

        Notes
        -----
        This method always synchronizes the stream before returning,
        ensuring outputs are ready to read on the host.

        """
        verbose = verbose if verbose is not None else self._verbose
        if verbose:
            LOG.info(f"{time.perf_counter()} {self.name} Dispatch: BEGIN")

        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["execute"])

        with self._device_guard:
            # reset the input bindings if direct_exec or raw_exec were used
            if not self._using_engine_tensors:
                self._set_input_bindings()
                self._using_engine_tensors = True

            # adjust the context shapes for dynamic engines so partial batches
            # only compute and copy at the submitted size, not the max profile
            if self._has_dynamic_inputs:
                self._apply_input_shapes(data)
            partial_shapes = self._active_output_shapes is not None

            # copy inputs
            for i_idx in range(len(self._inputs)):
                self._inputs[i_idx].upload(
                    data[i_idx],
                    self._stream,
                    self._active_input_shapes[i_idx] if partial_shapes else None,
                )

            if debug:
                stream_synchronize(self._stream)

            # execute
            # CUDA graphs are captured at the allocated (max) shapes; when the
            # context is at a smaller dynamic shape, fall through to a plain
            # enqueue for this call. The captured graph stays valid for future
            # calls at the allocated shapes.
            if self._cuda_graph and not partial_shapes:
                if self._cuda_graph.is_captured:
                    # uses already captured graph to handle execution
                    self._cuda_graph.launch()
                elif not self._capturing_graph:
                    # Capture the graph (warmup inside will use random data)
                    self._capture_cuda_graph()
                    # After capture, re-copy user's input (warmup overwrote it) and launch
                    if self._cuda_graph is not None and self._cuda_graph.is_captured:
                        for i_idx in range(len(self._inputs)):
                            self._inputs[i_idx].upload(data[i_idx], self._stream)
                        self._cuda_graph.launch()
                else:
                    # Currently capturing graph, use direct execution for warmup
                    self._context.execute_async_v3(self._stream)
            # base execution cases
            elif self._async_v3:
                self._context.execute_async_v3(self._stream)
            else:
                self._context.execute_async_v2(self._allocations, self._stream)

            if debug:
                stream_synchronize(self._stream)

            # copy outputs
            for o_idx in range(len(self._outputs)):
                self._outputs[o_idx].download(
                    self._stream,
                    self._active_output_shapes[o_idx]
                    if partial_shapes and self._active_output_shapes is not None
                    else None,
                )

            # make sure all operations are complete
            # Skip sync when warming up for graph capture to avoid conflicts
            # with cudaStreamCaptureModeGlobal in multi-threaded scenarios
            if not self._capturing_graph:
                stream_synchronize(self._stream)

        if verbose:
            LOG.info(f"{time.perf_counter()} {self.name} Dispatch: END")

        # return the results
        if (
            partial_shapes
            and self._active_output_shapes is not None
            and self._active_output_sizes is not None
        ):
            # only the prefix of each host allocation is valid, return views
            # (or copies) shaped to the actual context output shapes
            outputs = []
            for o_idx, o_binding in enumerate(self._outputs):
                o_shape = self._active_output_shapes[o_idx]
                size = self._active_output_sizes[o_idx]
                arr = o_binding.host_allocation.reshape(-1)[:size].reshape(o_shape)
                outputs.append(arr if no_copy else arr.copy())
        elif no_copy:
            outputs = [o.host_allocation for o in self._outputs]
        else:
            outputs = [o.host_allocation.copy() for o in self._outputs]

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()

        return outputs

    def graph_exec(
        self: Self,
        *,
        debug: bool | None = None,
    ) -> None:
        """
        Launch the captured CUDA graph.

        This method only launches the graph - it does not handle
        input/output memory transfers or graph capture. The graph must
        already be captured (via warmup or prior execute() calls).

        This method does NOT synchronize the stream by default, allowing
        the graph to be embedded in a larger pipeline. Use debug=True
        to force synchronization.

        Parameters
        ----------
        debug : bool, optional
            If True, synchronize the stream after graph launch.
            By default False (no synchronization).

        Raises
        ------
        RuntimeError
            If no CUDA graph has been captured or CUDA graphs are disabled.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["graph_exec"])

        with self._device_guard:
            if self._cuda_graph is None or not self._cuda_graph.is_captured:
                err_msg = f"No CUDA graph captured for engine '{self._name}'. "
                err_msg += "Ensure cuda_graph=True and warmup=True, or call execute() first."
                if FLAGS.NVTX_ENABLED:
                    nvtx.pop_range()  # graph_exec
                raise RuntimeError(err_msg)
            self._cuda_graph.launch()
            if debug:
                stream_synchronize(self._stream)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()

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

        Notes
        -----
        This method always synchronizes the stream before returning,
        ensuring outputs are ready to read on the host.

        """
        verbose = verbose if verbose is not None else self._verbose
        if not no_warn:
            LOG.warning(
                "Calling direct_exec is potentially dangerous, ensure all pointers and data are valid. Outputs can be overwritten inplace!",
            )

        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["direct_exec"])

        with self._device_guard:
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
            # the context may sit at a smaller dynamic shape than the
            # allocations, in which case only the valid prefix of each output
            # is copied back and returned. callers resolve the shape with
            # _resolve_dynamic_batch before dispatching here.
            partial_shapes = self._active_output_shapes is not None
            for o_idx in range(len(self._outputs)):
                self._outputs[o_idx].download(
                    self._stream,
                    self._active_output_shapes[o_idx]
                    if partial_shapes and self._active_output_shapes is not None
                    else None,
                )

            # make sure all operations are complete
            stream_synchronize(self._stream)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()

        # return the output host allocations, shaped to the active context
        if (
            partial_shapes
            and self._active_output_shapes is not None
            and self._active_output_sizes is not None
        ):
            return [
                o_binding.host_allocation.reshape(-1)[: self._active_output_sizes[o_idx]].reshape(
                    self._active_output_shapes[o_idx]
                )
                for o_idx, o_binding in enumerate(self._outputs)
            ]
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

        Notes
        -----
        This method does NOT synchronize the stream by default. The caller
        is responsible for synchronization if needed. Use debug=True to
        force synchronization after execution.

        """
        verbose = verbose if verbose is not None else self._verbose
        if not no_warn:
            LOG.warning(
                "Calling raw_exec is potentially dangerous, ensure all pointers and data are valid. Outputs can be overwritten inplace!",
            )

        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["raw_exec"])

        with self._device_guard:
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

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()

        # return the pointers to the output allocations
        return self._output_allocations
