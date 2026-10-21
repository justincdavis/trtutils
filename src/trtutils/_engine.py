# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

import nvtx

from ._flags import FLAGS
from ._log import LOG
from .core._buffer import Buffer, as_buffers
from .core._graph import CUDAGraph
from .core._interface import TRTEngineInterface
from .core._stream import stream_synchronize

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import ClassVar

    import numpy as np
    from typing_extensions import Self

    from trtutils.compat._libs import cuda

    # the shape and device address of every input
    _GraphKey = tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]


# the most CUDA graphs (and seen-once configurations) an engine keeps
_MAX_CUDA_GRAPHS = 8


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

        self._init_io_state(cuda_graph=cuda_graph)

        # store timing variable for sleep call before stream_sync
        self._sync_t: float = 0.0

        # store verbose info
        self._verbose = verbose if verbose is not None else False

        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["warmup"])

        if warmup:
            self.warmup(warmup_iterations, verbose=self._verbose)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # warmup
            nvtx.pop_range()  # init

        LOG.debug(f"Created TRTEngine: {self.name}")

    def _init_io_state(self: Self, *, cuda_graph: bool | None) -> None:
        """Point the context at the bindings at their max shapes and reset the graph cache."""
        # CUDA graph support: one graph per input configuration (the shape and
        # device address of every input), least recently used first. A graph
        # can only replay on a call that does not change the shapes, so a
        # configuration counts only when it runs at unchanged shapes, and is
        # captured the second time it does; configurations seen once are
        # tracked separately so inputs that never repeat (or alternate shapes
        # every call) neither pay for a capture nor evict graphs worth keeping.
        self._cuda_graph_enabled: bool = (
            cuda_graph if cuda_graph is not None else True
        ) and self._async_v3
        self._cuda_graphs: OrderedDict[_GraphKey, CUDAGraph] = OrderedDict()
        self._seen_configs: OrderedDict[_GraphKey, None] = OrderedDict()

        # the shapes and device addresses the context currently holds for each
        # input; every call compares against these, so the context is only
        # updated when a submitted Buffer differs from the previous call
        self._input_shapes: list[tuple[int, ...]] = [b.max_shape for b in self._inputs]
        self._input_addresses: list[int] = [b.allocation for b in self._inputs]
        # the Buffers the context reads, kept alive for graph_exec / raw_exec,
        # which use them after the call that bound them returns
        self._bound_inputs: list[Buffer] = [b.device for b in self._inputs]
        self._output_shapes: list[tuple[int, ...]] = [b.max_shape for b in self._outputs]
        # set when the context shapes change; the next enqueue then runs
        # without a CUDA graph so TensorRT can react to the new shapes
        self._shapes_changed: bool = False
        if self._async_v3:
            for i_binding in self._inputs:
                self._context.set_input_shape(i_binding.name, i_binding.shape)
                self._context.set_tensor_address(i_binding.name, i_binding.allocation)
            for o_binding in self._outputs:
                self._context.set_tensor_address(o_binding.name, o_binding.allocation)

    @property
    def active_input_shapes(self: Self) -> list[tuple[int, ...]]:
        """The input shapes of the most recent execution."""
        return list(self._input_shapes)

    @property
    def active_output_shapes(self: Self) -> list[tuple[int, ...]]:
        """The output shapes of the most recent execution."""
        return list(self._output_shapes)

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
                shape = o_binding.max_shape
            shapes.append(shape)
        self._output_shapes = shapes

    def _set_input_address(self: Self, index: int, address: int) -> None:
        if address == self._input_addresses[index]:
            return
        binding = self._inputs[index]
        if self._async_v3:
            self._context.set_tensor_address(binding.name, address)
        else:
            # the v2 backend takes a pointer per binding index instead of names
            self._allocations[binding.index] = address
        self._input_addresses[index] = address

    def _validate_inputs(self: Self, data: Sequence[Buffer]) -> list[Buffer]:
        """
        Check the inputs against the engine without touching the context.

        Raises
        ------
        ValueError
            If the number of inputs, a dtype, or a shape does not match the engine.

        """
        inputs = as_buffers(data)
        if len(inputs) != len(self._inputs):
            err_msg = f"Engine '{self._name}' expects {len(self._inputs)} inputs, got {len(inputs)}."
            raise ValueError(err_msg)
        for binding, buffer in zip(self._inputs, inputs):
            binding.check(buffer, self._name)
        return inputs

    def _bind_inputs(self: Self, data: Sequence[Buffer]) -> None:
        """
        Point the context at the given inputs, updating shapes and addresses as needed.

        Every input is validated before the context is touched, so a bad
        input never leaves the engine half-updated. Device Buffers (and
        mapped host Buffers) are bound in place; host Buffers, and device
        Buffers TensorRT cannot address directly, are copied into the
        engine's own input bindings on the engine stream.
        """
        inputs = self._validate_inputs(data)
        shapes_changed = False
        try:
            for index, (binding, buffer) in enumerate(zip(self._inputs, inputs)):
                if buffer.shape != self._input_shapes[index]:
                    self._set_context_shape(index, buffer.shape)
                    shapes_changed = True
                bound = binding.bind(buffer, self._stream)
                self._set_input_address(index, bound.device_ptr)
                self._bound_inputs[index] = bound
        finally:
            # even if a later input fails, the outputs must describe the
            # shapes the context now holds
            if shapes_changed:
                self._refresh_output_shapes()

    def _graph_key(self: Self) -> _GraphKey:
        """Get the input configuration a CUDA graph is captured for and replays."""
        return tuple(self._input_shapes), tuple(self._input_addresses)

    def _current_graph(self: Self) -> CUDAGraph | None:
        """Get the captured CUDA graph for the context's current input configuration."""
        return self._cuda_graphs.get(self._graph_key())

    def _enqueue(self: Self, *, allow_graph: bool) -> None:
        """Enqueue one execution of the context on the engine stream."""
        if not self._async_v3:
            self._check_enqueued(self._context.execute_async_v2(self._allocations, self._stream))
            return
        key = self._graph_key() if allow_graph and self._cuda_graph_enabled else None
        graph = self._cuda_graphs.get(key) if key is not None else None
        # right after a shape change TensorRT needs one real enqueue before
        # a graph for the new shapes may be captured or replayed
        if key is not None and graph is not None:
            self._cuda_graphs.move_to_end(key)
            if not self._shapes_changed:
                graph.launch()
                return
        self._check_enqueued(self._context.execute_async_v3(self._stream))
        shapes_changed, self._shapes_changed = self._shapes_changed, False
        if key is None or graph is not None or shapes_changed:
            return
        if key in self._seen_configs:
            # second run at unchanged shapes: record it for the next calls
            del self._seen_configs[key]
            graph = self._capture_cuda_graph()
            if graph is not None:
                self._store_graph(key, graph)
        else:
            self._seen_configs[key] = None
            if len(self._seen_configs) > _MAX_CUDA_GRAPHS:
                self._seen_configs.popitem(last=False)

    def _check_enqueued(self: Self, ok: object) -> None:
        if ok is False:
            err_msg = (
                f"TensorRT failed to enqueue engine '{self._name}' at input shapes "
                f"{self._input_shapes}; check that the shapes are consistent with each other."
            )
            raise RuntimeError(err_msg)

    def _store_graph(self: Self, key: _GraphKey, graph: CUDAGraph) -> None:
        """Cache a captured graph, invalidating the least recently used beyond the limit."""
        self._cuda_graphs[key] = graph
        while len(self._cuda_graphs) > _MAX_CUDA_GRAPHS:
            _, evicted = self._cuda_graphs.popitem(last=False)
            evicted.invalidate()

    def _clear_graphs(self: Self) -> None:
        for graph in self._cuda_graphs.values():
            graph.invalidate()
        self._cuda_graphs.clear()
        self._seen_configs.clear()

    def _capture_cuda_graph(self: Self) -> CUDAGraph | None:
        """
        Record one execution of the context at its current configuration.

        The call that triggers a capture has already run for real, so a
        failed capture only logs a warning and disables CUDA graphs for the
        engine; execution carries on without them.

        Returns
        -------
        CUDAGraph | None
            The captured graph, or None if capture failed.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["graph_capture"])
        graph = CUDAGraph(self._stream)
        # serialize CUDA graph capture across engines
        with self._capture_lock, graph:
            enqueued = self._context.execute_async_v3(self._stream)
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # graph_capture
        if enqueued is False or not graph.is_captured:
            if graph.is_captured:
                # a rejected enqueue still records a graph, which must never replay
                graph.invalidate()
            self._cuda_graph_enabled = False
            self._clear_graphs()
            LOG.warning(
                f"CUDA graph capture failed for engine '{self._name}'; continuing without "
                "CUDA graphs. Pass cuda_graph=False to skip the capture attempt."
            )
            return None
        return graph

    def __del__(self: Self) -> None:
        with contextlib.suppress(AttributeError):
            self._clear_graphs()
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
        try:
            with self._device_guard:
                self._run(data, allow_graph=True, debug=debug)
                outputs = self._fetch_outputs()
                stream_synchronize(self._stream)
        finally:
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()  # execute

        if verbose:
            LOG.info(f"{time.perf_counter()} {self.name} Dispatch: END")

        return [o.array if no_copy else o.array.copy() for o in outputs]

    def _run(self: Self, data: Sequence[Buffer], *, allow_graph: bool, debug: bool | None) -> None:
        """Bind the inputs and enqueue one execution on the engine stream."""
        self._bind_inputs(data)
        if debug:
            stream_synchronize(self._stream)
        self._enqueue(allow_graph=allow_graph)
        if debug:
            stream_synchronize(self._stream)

    def _fetch_outputs(self: Self) -> list[Buffer]:
        """Enqueue copies of the latest outputs into the host buffers of the output bindings."""
        return [
            binding.fetch(shape, self._stream)
            for binding, shape in zip(self._outputs, self._output_shapes)
        ]

    def graph_exec(
        self: Self,
        *,
        debug: bool | None = None,
    ) -> None:
        """
        Launch the CUDA graph captured for the engine's current input configuration.

        This method only launches the graph - it does not handle
        input/output memory transfers or graph capture. A graph is captured
        once an input configuration (the shape and device address of every
        input) has been executed twice at unchanged shapes, e.g. by warmup or
        prior execute() calls; the graph launched here replays the
        configuration of the most recent execution.

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
            graph = self._current_graph()
            if graph is None:
                err_msg = f"No CUDA graph captured for engine '{self._name}'. "
                err_msg += "Ensure cuda_graph=True and execute the current inputs twice at "
                err_msg += "unchanged shapes first (warmup iterations count)."
                if FLAGS.NVTX_ENABLED:
                    nvtx.pop_range()  # graph_exec
                raise RuntimeError(err_msg)
            graph.launch()
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
        Buffers must stay unmodified, and the outputs must not be read,
        until the caller synchronizes the stream.

        """
        verbose = verbose if verbose is not None else self._verbose
        if verbose:
            LOG.info(f"{time.perf_counter()} {self.name} raw_exec")

        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["raw_exec"])
        try:
            with self._device_guard:
                self._run(data, allow_graph=False, debug=debug)
        finally:
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()  # raw_exec

        # fresh views, so a caller freeing an output never frees the binding
        return [b.device.view(s) for b, s in zip(self._outputs, self._output_shapes)]
