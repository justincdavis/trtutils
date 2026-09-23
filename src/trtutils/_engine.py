# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import nvtx

from ._flags import FLAGS
from ._log import LOG
from .core._buffer import DEVICE_ALIGNMENT, Buffer, as_buffers
from .core._graph import CUDAGraph
from .core._interface import TRTEngineInterface
from .core._stream import stream_synchronize

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import ClassVar

    import numpy as np
    from typing_extensions import Self

    from trtutils.compat._libs import cuda


def _bindable(buffer: Buffer) -> bool:
    """Whether TensorRT can read a Buffer in place (device-visible and aligned)."""
    return buffer.device_visible and buffer.device_ptr % DEVICE_ALIGNMENT == 0


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
        if self._cuda_graph_enabled:
            self._cuda_graph = CUDAGraph(self._stream)

        # the shapes and device addresses the context currently holds for each
        # input; every call compares against these, so the context is only
        # updated when a submitted Buffer differs from the previous call
        self._input_shapes: list[tuple[int, ...]] = [tuple(b.shape) for b in self._inputs]
        self._input_addresses: list[int] = [b.allocation for b in self._inputs]
        self._output_shapes: list[tuple[int, ...]] = [tuple(b.shape) for b in self._outputs]
        # set when the context shapes change; the next enqueue then runs
        # without a CUDA graph so TensorRT can react to the new shapes
        self._shapes_changed: bool = False
        # the v2 backend takes a pointer per binding index instead of names
        self._v2_pointers: list[int] = list(self._allocations)
        if self._async_v3:
            for i_binding in self._inputs:
                self._context.set_input_shape(i_binding.name, i_binding.shape)
                self._context.set_tensor_address(i_binding.name, i_binding.allocation)
            for o_binding in self._outputs:
                self._context.set_tensor_address(o_binding.name, o_binding.allocation)

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

    @property
    def active_input_shapes(self: Self) -> list[tuple[int, ...]]:
        """The input shapes of the most recent execution."""
        return list(self._input_shapes)

    @property
    def active_output_shapes(self: Self) -> list[tuple[int, ...]]:
        """The output shapes of the most recent execution."""
        return list(self._output_shapes)

    def _check_input_shape(self: Self, index: int, shape: tuple[int, ...]) -> None:
        name = self._inputs[index].name
        min_shape = self._input_min_shapes[index]
        max_shape = self._input_max_shapes[index]
        valid = len(shape) == len(min_shape) and all(
            lo <= dim <= hi for dim, lo, hi in zip(shape, min_shape, max_shape)
        )
        if not valid:
            if min_shape == max_shape:
                expected = f"exactly {min_shape}"
            else:
                expected = f"between {min_shape} and {max_shape}"
            err_msg = (
                f"Input '{name}' of engine '{self._name}' got shape {shape}, expected {expected}."
            )
            raise ValueError(err_msg)

    def _set_context_shape(self: Self, index: int, shape: tuple[int, ...]) -> None:
        binding = self._inputs[index]
        if self._async_v3:
            ok = self._context.set_input_shape(binding.name, shape)
        else:
            ok = self._context.set_binding_shape(binding.index, shape)
        if ok is False:
            err_msg = f"TensorRT rejected shape {shape} for input '{binding.name}' of engine '{self._name}'."
            raise ValueError(err_msg)
        self._input_shapes[index] = shape
        self._shapes_changed = True

    def _refresh_output_shapes(self: Self) -> None:
        shapes: list[tuple[int, ...]] = []
        for o_binding in self._outputs:
            if self._async_v3:
                shape = tuple(self._context.get_tensor_shape(o_binding.name))
            else:
                shape = tuple(self._context.get_binding_shape(o_binding.index))
            # data-dependent output shapes are only known after execution;
            # fall back to the full allocation for those
            if any(dim < 0 for dim in shape):
                shape = tuple(o_binding.shape)
            shapes.append(shape)
        self._output_shapes = shapes

    def _set_input_address(self: Self, index: int, address: int) -> None:
        if address == self._input_addresses[index]:
            return
        binding = self._inputs[index]
        if self._async_v3:
            self._context.set_tensor_address(binding.name, address)
        else:
            self._v2_pointers[binding.index] = address
        self._input_addresses[index] = address

    def _bind_inputs(self: Self, data: Sequence[Buffer]) -> None:
        """
        Point the context at the given inputs, updating shapes and addresses as needed.

        Every input is validated before the context is touched, so a bad
        input never leaves the engine half-updated. Device Buffers (and
        mapped host Buffers) are bound in place; host Buffers, and device
        Buffers TensorRT cannot address directly, are copied into the
        engine's own input bindings on the engine stream.

        Raises
        ------
        ValueError
            If the number of inputs, a dtype, or a shape does not match the engine.

        """
        inputs = as_buffers(data)
        if len(inputs) != len(self._inputs):
            err_msg = f"Engine '{self._name}' expects {len(self._inputs)} inputs, got {len(inputs)}."
            raise ValueError(err_msg)
        for index, (binding, buffer) in enumerate(zip(self._inputs, inputs)):
            if buffer.dtype != binding.dtype:
                err_msg = (
                    f"Input '{binding.name}' of engine '{self._name}' expects dtype "
                    f"{binding.dtype}, got {buffer.dtype}."
                )
                raise ValueError(err_msg)
            if buffer.shape != self._input_shapes[index]:
                self._check_input_shape(index, buffer.shape)

        shapes_changed = False
        for index, (binding, buffer) in enumerate(zip(self._inputs, inputs)):
            if buffer.shape != self._input_shapes[index]:
                self._set_context_shape(index, buffer.shape)
                shapes_changed = True
            if _bindable(buffer):
                address = buffer.device_ptr
            else:
                address = binding.stage(buffer, self._stream).ptr
            self._set_input_address(index, address)
        if shapes_changed:
            self._refresh_output_shapes()

    @property
    def _at_allocated_bindings(self: Self) -> bool:
        """Whether the context holds exactly the engine's own bindings at their full shapes."""
        return all(
            address == binding.allocation and shape == tuple(binding.shape)
            for binding, address, shape in zip(
                self._inputs, self._input_addresses, self._input_shapes
            )
        )

    def _enqueue(self: Self, *, allow_graph: bool) -> None:
        """Enqueue one execution of the context on the engine stream."""
        if not self._async_v3:
            self._context.execute_async_v2(self._v2_pointers, self._stream)
            return
        # the CUDA graph was captured against the engine's own bindings at
        # their full shapes, so it only replays for exactly that configuration
        use_graph = (
            allow_graph
            and self._cuda_graph is not None
            and not self._shapes_changed
            and self._at_allocated_bindings
        )
        if use_graph and self._cuda_graph is not None and self._cuda_graph.is_captured:
            self._cuda_graph.launch()
            return
        self._context.execute_async_v3(self._stream)
        self._shapes_changed = False
        if use_graph:
            # this call already ran for real; record a graph for the next ones
            self._capture_cuda_graph()

    def _capture_cuda_graph(self: Self) -> None:
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["graph_capture"])
        graph = self._cuda_graph
        if graph is None:
            err_msg = f"CUDA graph is not enabled in engine: {self._name}"
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()  # graph_capture
            raise RuntimeError(err_msg)
        # serialize CUDA graph capture across engines
        with self._capture_lock, graph:
            self._context.execute_async_v3(self._stream)
        if not graph.is_captured:
            self._cuda_graph = None
            err_msg = (
                f"CUDA graph capture failed for engine '{self._name}'.\n"
                "The engine may not support CUDA graph capture.\n"
                "To resolve: use cuda_graph=False to disable CUDA graphs for this engine."
            )
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()  # graph_capture
            raise RuntimeError(err_msg)
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # graph_capture

    def __del__(self: Self) -> None:
        with contextlib.suppress(AttributeError):
            if self._cuda_graph is not None:
                self._cuda_graph.invalidate()
        super().__del__()

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

        Each input is a :class:`~trtutils.core.Buffer` on the host or the
        device. The engine runs at the shapes of the given Buffers, so a
        dynamic engine executes exactly the submitted batch/resolution and
        returns outputs of the matching shape. Device Buffers are read in
        place (no copy); host Buffers are copied to the device first.

        Parameters
        ----------
        data : Sequence[Buffer]
            One Buffer per engine input. Wrap numpy arrays or CUDA arrays
            (CuPy, PyTorch, ...) with :meth:`Buffer.wrap`.
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
            try:
                self._bind_inputs(data)
            except (TypeError, ValueError):
                if FLAGS.NVTX_ENABLED:
                    nvtx.pop_range()  # execute
                raise

            if debug:
                stream_synchronize(self._stream)

            self._enqueue(allow_graph=True)

            if debug:
                stream_synchronize(self._stream)

            outputs = [
                binding.fetch(shape, self._stream)
                for binding, shape in zip(self._outputs, self._output_shapes)
            ]
            stream_synchronize(self._stream)

        if verbose:
            LOG.info(f"{time.perf_counter()} {self.name} Dispatch: END")

        results = [o.array if no_copy else o.array.copy() for o in outputs]

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()

        return results

    def stage_inputs(self: Self, data: Sequence[Buffer]) -> list[Buffer]:
        """
        Make every input device-resident, copying host Buffers into the engine's bindings.

        Useful before recording the engine into an outer CUDA graph with
        :meth:`raw_exec`, since host-to-device copies from pageable memory
        cannot be captured. The copies are enqueued on the engine stream.

        Parameters
        ----------
        data : Sequence[Buffer]
            One Buffer per engine input.

        Returns
        -------
        list[Buffer]
            Device Buffers holding the inputs, in the same order.

        """
        staged: list[Buffer] = []
        with self._device_guard:
            for binding, buffer in zip(self._inputs, as_buffers(data)):
                if _bindable(buffer):
                    staged.append(buffer)
                else:
                    staged.append(binding.stage(buffer, self._stream))
        return staged

    def graph_exec(
        self: Self,
        *,
        debug: bool | None = None,
    ) -> None:
        """
        Launch the captured CUDA graph.

        This method only launches the graph - it does not handle
        input/output memory transfers or graph capture. The graph must
        already be captured (via warmup or prior execute() calls) and
        replays the engine's own input/output bindings at their full shapes.

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

    def raw_exec(
        self: Self,
        data: Sequence[Buffer],
        *,
        verbose: bool | None = None,
        debug: bool | None = None,
    ) -> list[Buffer]:
        """
        Enqueue the network on its stream, leaving the outputs on the device.

        Inputs are handled as in :meth:`execute`. The engine's own CUDA
        graph is never used, so this call can be recorded into an outer
        graph. The returned Buffers view the engine's output bindings and
        are overwritten by the next execution.

        Parameters
        ----------
        data : Sequence[Buffer]
            One Buffer per engine input.
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

        Notes
        -----
        This method does NOT synchronize the stream by default. Input
        Buffers must stay alive and unmodified, and the outputs must not be
        read, until the caller synchronizes the stream.

        """
        verbose = verbose if verbose is not None else self._verbose
        if verbose:
            LOG.info(f"{time.perf_counter()} {self.name} raw_exec")

        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["raw_exec"])

        with self._device_guard:
            try:
                self._bind_inputs(data)
            except (TypeError, ValueError):
                if FLAGS.NVTX_ENABLED:
                    nvtx.pop_range()  # raw_exec
                raise
            self._enqueue(allow_graph=False)
            if debug:
                stream_synchronize(self._stream)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()

        return [
            binding.device.view(shape) for binding, shape in zip(self._outputs, self._output_shapes)
        ]
