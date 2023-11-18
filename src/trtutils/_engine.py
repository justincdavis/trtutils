from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

if TYPE_CHECKING:
    from typing_extensions import Self


class TRTEngine:
    """
    A wrapper around a TensorRT engine that handles the device memory.

    It is thread and process safe to create multiple TRTEngines.
    It is valid to create a TRTEngine in one thread and use in another.
    Each TRTEngine has its own CUDA context and there is no safeguards
    implemented in the class for datarace conditions. As such, a
    single TRTEngine should not be used in multiple threads or processes.

    Attributes
    ----------
    input_shapes : list[tuple[int, ...]]
        The shapes of the inputs

    Methods
    -------
    __call__(inputs: list[np.ndarray])
        Execute the engine with the given inputs
    execute(inputs: list[np.ndarray])
        Execute the engine with the given inputs
    mock_execute()
        Execute the engine with random inputs
    """

    def __init__(
        self: Self,
        engine_path: str,
        warmup: bool | None = None,
        warmup_iterations: int = 5,
        dtype: np.number = np.float32,
        device: int = 0,
    ) -> None:
        """
        Use to initialize the TRTEngine.

        Parameters
        ----------
        engine_path : str
            The path to the serialized engine file
        warmup : bool, optional
            Whether to do warmup iterations, by default None
            If None, warmup will be set to False
        warmup_iterations : int, optional
            The number of warmup iterations to do, by default 5
        dtype : np.number, optional
            The datatype to use for the inputs and outputs, by default np.float32
        device : int, optional
            The device to use, by default 0
        """
        # get a unique context for thread safe operation
        self._cfx = cuda.Device(device).make_context()

        # set the datatype
        self._dtype = dtype

        # load the libnvinfer plugins
        trt.init_libnvinfer_plugins(None, "")

        # load the engine from file
        self._trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self._trt_logger) as runtime:
            self._engine = runtime.deserialize_cuda_engine(f.read())

        if self._engine is None:
            raise RuntimeError("Could not serialize engine")

        # create the execution context
        self._context = self._engine.create_execution_context()
        self._context.active_optimization_profile = 0

        # get the binding idxs
        self._input_binding_idxs, self._output_binding_idxs = self._get_binding_idxs()
        self._input_names = [
            self._engine.get_binding_name(binding_idx)
            for binding_idx in self._input_binding_idxs
        ]

        # generate the random inputs
        self._host_inputs, self._input_shapes = self._get_random_inputs(seed=1)

        # Allocate device memory for inputs. This can be easily re-used if the
        # input shapes don't change
        self._device_inputs = [
            cuda.mem_alloc(h_input.nbytes) for h_input in self._host_inputs
        ]
        # Copy host inputs to device, this needs to be done for each new input
        for h_input, d_input in zip(self._host_inputs, self._device_inputs):
            cuda.memcpy_htod(d_input, h_input)

        # allocate the outputs
        self._host_outputs, self._device_outputs = self._setup_binding_shapes()
        self._output_names = [
            self._engine.get_binding_name(binding_idx)
            for binding_idx in self._output_binding_idxs
        ]

        # Bindings are a list of device pointers for inputs and outputs
        self._bindings = self._device_inputs + self._device_outputs

        # do any warmup iterations
        if warmup is None:
            warmup = False
        if warmup:
            for _ in range(warmup_iterations):
                self.mock_execute()

    @property
    def input_shapes(self: Self) -> list[tuple[int, ...]]:
        """
        The shapes of the inputs.

        Returns
        -------
        list[tuple[int, ...]]
            The shapes of the inputs
        """
        return self._input_shapes

    def __del__(self: Self) -> None:
        def _del(obj: object, attr: str) -> None:
            with contextlib.suppress(AttributeError):
                delattr(obj, attr)

        with contextlib.suppress(AttributeError):
            self._cfx.pop()

        attrs = ["_tegra", "_cfx", "_context", "_engine"]
        for attr in attrs:
            _del(self, attr)

    def _get_binding_idxs(self: Self) -> tuple[list[int], list[int]]:
        # get the profile_index
        profile_index = self._context.active_optimization_profile

        # Calculate start/end binding indices for current context's profile
        num_bindings_per_profile = (
            self._engine.num_bindings // self._engine.num_optimization_profiles
        )
        start_binding = profile_index * num_bindings_per_profile
        end_binding = start_binding + num_bindings_per_profile

        # Separate input and output binding indices for convenience
        input_binding_idxs = []
        output_binding_idxs = []
        for binding_index in range(start_binding, end_binding):
            if self._engine.binding_is_input(binding_index):
                input_binding_idxs.append(binding_index)
            else:
                output_binding_idxs.append(binding_index)

        return input_binding_idxs, output_binding_idxs

    def _get_random_inputs(
        self: Self,
    ) -> tuple[list[np.ndarray], list[tuple[int, ...]]]:
        # Input data for inference
        host_inputs = []
        input_shapes = []

        for binding_index in self._input_binding_idxs:
            # If input shape is fixed, we'll just use it
            input_shape = self._context.get_binding_shape(binding_index)
            # If input shape is dynamic, we'll arbitrarily select one of the
            # the min/opt/max shapes from our optimization profile
            if self._is_dynamic(input_shape):
                profile_index = self._context.active_optimization_profile
                profile_shapes = self._engine.get_profile_shape(
                    profile_index, binding_index
                )
                # 0=min, 1=opt, 2=max, or choose any shape, (min <= shape <= max)
                input_shape = profile_shapes[1]

            host_inputs.append(np.random.random(input_shape).astype(self._dtype))
            input_shapes.append(input_shape)

        return host_inputs, input_shapes

    def _setup_binding_shapes(self: Self) -> tuple[list[np.ndarray], list[int]]:
        # Explicitly set the dynamic input shapes, so the dynamic output
        # shapes can be computed internally
        for host_input, binding_index in zip(
            self._host_inputs, self._input_binding_idxs
        ):
            self._context.set_binding_shape(binding_index, host_input.shape)

        assert self._context.all_binding_shapes_specified

        host_outputs = []
        device_outputs = []
        for binding_index in self._output_binding_idxs:
            output_shape = self._context.get_binding_shape(binding_index)
            # Allocate buffers to hold output results after copying back to host
            buffer = np.empty(output_shape, dtype=self._dtype)
            host_outputs.append(buffer)
            # Allocate output buffers on device
            device_outputs.append(cuda.mem_alloc(buffer.nbytes))

        return host_outputs, device_outputs

    def _is_fixed(self: Self, shape: tuple[int]) -> bool:
        return not self._is_dynamic(shape)

    @staticmethod
    def _is_dynamic(shape: tuple[int]) -> bool:
        return any(dim is None or dim < 0 for dim in shape)

    def __call__(self: Self, inputs: list[np.ndarray]) -> Any:  # noqa: ANN401
        """
        Execute the engine with the given inputs.

        Parameters
        ----------
        inputs : list[np.ndarray]
            The inputs to the engine


        Returns
        -------
        Any
            The outputs from the engine
        """
        return self.execute(inputs)

    def execute(self: Self, inputs: list[np.ndarray]) -> np.ndarray:
        """
        Execute the engine with the given inputs.

        Parameters
        ----------
        inputs : list[np.ndarray]
            The inputs to the engine


        Returns
        -------
        np.ndarray
            The outputs from the engine
        """
        # Get context
        self._cfx.push()
        # Copy data to the device
        for h_input, d_input in zip(inputs, self._device_inputs):
            cuda.memcpy_htod(d_input, h_input)
        # Bindings are a list of device pointers for inputs and outputs
        bindings = self._device_inputs + self._device_outputs
        # Inference
        self._context.execute_v2(bindings)
        # Copy outputs back to host to view results
        for h_output, d_output in zip(self._host_outputs, self._device_outputs):
            cuda.memcpy_dtoh(h_output, d_output)
        # Release context
        self._cfx.pop()
        return self._host_outputs

    def mock_execute(self: Self) -> np.ndarray:
        """
        Execute the engine with random inputs.

        Returns
        -------
        np.ndarray
            The outputs from the engine
        """
        mock_inputs = [
            np.random.random(s).astype(self._dtype) for s in self.input_shapes
        ]
        return self.execute(mock_inputs)
