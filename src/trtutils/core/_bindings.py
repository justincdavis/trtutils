from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import tensorrt as trt
from cuda import cudart

from ._cuda import cuda_call

if TYPE_CHECKING:
    from typing_extensions import Self


@dataclass
class Binding:
    index: int
    name: str
    dtype: np.dtype
    shape: list[int]
    is_input: bool
    allocation: int
    host_allocation: np.ndarray | None

    def free(self: Self) -> None:
        """
        Free the memory of the binding.
        
        Raises
        ------
        RuntimeError
            If a CUDA error occurred.
        
        """
        cuda_call(cudart.cudaFree(self.allocation))


def allocate_bindings(
    engine: trt.IEngine,
    context: trt.IExecutionContext,
    logger: trt.ILogger,
) -> tuple[list[Binding], list[Binding], list[int], int]:
    """
    Allocate memory for the input and output tensors of a TensorRT engine.

    Parameters
    ----------
    engine : trt.IEngine
        The TensorRT engine to allocate memory for.
    context : trt.IExecutionContext
        The execution context to use.
    logger : trt.ILogger
        The logger to use.

    Returns
    -------
    tuple[list[Binding], list[Binding], list[int], int]
        A tuple containing the input bindings, output bindings, allocations, and batch size.

    Raises
    ------
    ValueError
        If the batch size is 0.
        If no input tensors are found.
        If no output tensors are found.
        If no memory allocations are found

    """
    inputs = []
    outputs = []
    allocations = []
    batch_size = 0

    # based on the version of tensorrt, num_io_tensors is not available in IEngine
    if int(trt.__version__.split(".")[1]) >= 5:
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            is_input = False
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
            shape = context.get_tensor_shape(name)
            if is_input and shape[0] < 0:
                assert engine.num_optimization_profiles > 0
                profile_shape = engine.get_tensor_profile_shape(name, 0)
                assert len(profile_shape) == 3  # min,opt,max
                # Set the *max* profile as binding shape
                context.set_input_shape(name, profile_shape[2])
                shape = context.get_tensor_shape(name)
            if is_input:
                batch_size = shape[0]
            size = dtype.itemsize
            for s in shape:
                size *= s
            allocation = cuda_call(cudart.cudaMalloc(size))
            host_allocation = None if is_input else np.zeros(shape, dtype)
            # binding = {
            #     "index": i,
            #     "name": name,
            #     "dtype": dtype,
            #     "shape": list(shape),
            #     "allocation": allocation,
            #     "host_allocation": host_allocation,
            # }
            binding = Binding(
                index=i,
                name=name,
                dtype=dtype,
                shape=list(shape),
                is_input=is_input,
                allocation=allocation,
                host_allocation=host_allocation,
            )
            allocations.append(allocation)
            if is_input:
                inputs.append(binding)
            else:
                outputs.append(binding)
            input_str = "Input" if is_input else "Output"
            logger.log(
                trt.Logger.INFO,
                f"{input_str} '{binding.name}' with shape {binding.shape} and dtype {binding.dtype}",
            )
    else:
        for i in range(engine.num_bindings):
            is_input = False
            if engine.binding_is_input(i):
                is_input = True
            name = engine.get_binding_name(i)
            dtype = np.dtype(trt.nptype(engine.get_binding_dtype(i)))
            shape = context.get_binding_shape(i)
            if is_input and shape[0] < 0:
                assert engine.num_optimization_profiles > 0
                profile_shape = engine.get_profile_shape(0, name)
                assert len(profile_shape) == 3  # min,opt,max
                # Set the *max* profile as binding shape
                context.set_binding_shape(i, profile_shape[2])
                shape = context.get_binding_shape(i)
            if is_input:
                batch_size = shape[0]
            size = dtype.itemsize
            for s in shape:
                size *= s
            allocation = cuda_call(cudart.cudaMalloc(size))
            host_allocation = None if is_input else np.zeros(shape, dtype)
            # binding = {
            #     "index": i,
            #     "name": name,
            #     "dtype": dtype,
            #     "shape": list(shape),
            #     "allocation": allocation,
            #     "host_allocation": host_allocation,
            # }
            binding = Binding(
                index=i,
                name=name,
                dtype=dtype,
                shape=list(shape),
                is_input=is_input,
                allocation=allocation,
                host_allocation=host_allocation,
            )
            allocations.append(allocation)
            if is_input:
                inputs.append(binding)
            else:
                outputs.append(binding)
            input_str = "Input" if is_input else "Output"
            logger.log(
                trt.Logger.INFO,
                f"{input_str} '{binding.name}' with shape {binding.shape} and dtype {binding.dtype}",
            )

    if batch_size == 0:
        err_msg = "Batch size is 0. Ensure that the engine has an input tensor with a valid batch size."
        raise ValueError(err_msg)
    if len(inputs) == 0:
        err_msg = "No input tensors found. Ensure that the engine has at least one input tensor."
        raise ValueError(err_msg)
    if len(outputs) == 0:
        err_msg = "No output tensors found. Ensure that the engine has at least one output tensor."
        raise ValueError(err_msg)
    if len(allocations) == 0:
        err_msg = "No memory allocations found. Ensure that the engine has at least one input and output tensor."
        raise ValueError(err_msg)

    return inputs, outputs, allocations, batch_size
