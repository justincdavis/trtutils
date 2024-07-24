#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import annotations

import ctypes

import numpy as np
import tensorrt as trt
from cuda import cuda, cudart

from ._cuda import cuda_call


class HostDeviceMem:
    """Pair of host and device memory, where the host memory is wrapped in a numpy array"""
    def __init__(self, size: int, dtype: np.dtype | None = None):
        dtype = dtype or np.dtype(np.uint8)
        nbytes = size * dtype.itemsize
        host_mem = cuda_call(cudart.cudaMallocHost(nbytes))
        pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))

        self._host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size,))
        self._device = cuda_call(cudart.cudaMalloc(nbytes))
        self._nbytes = nbytes

    @property
    def host(self) -> np.ndarray:
        return self._host

    @host.setter
    def host(self, data: np.ndarray | bytes):
        if isinstance(data, np.ndarray):
            if data.size > self.host.size:
                err_msg = f"Tried to fit an array of size {data.size} into host memory of size {self.host.size}"
                raise ValueError(err_msg)
            np.copyto(self.host[:data.size], data.flat, casting='safe')
        else:
            if not isinstance(self.host.dtype, np.uint8):
                err_msg = "Cannot copy bytes to non-uint8 array"
                raise ValueError(err_msg)
            self.host[:self.nbytes] = np.frombuffer(data, dtype=np.uint8)

    @property
    def device(self) -> int:
        return self._device

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def __str__(self):
        return f"Host:\n{self.host}\nDevice:\n{self.device}\nSize:\n{self.nbytes}\n"

    def __repr__(self):
        return self.__str__()

    def free(self):
        cuda_call(cudart.cudaFree(self.device))
        cuda_call(cudart.cudaFreeHost(self.host.ctypes.data))


# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
# If engine uses dynamic shapes, specify a profile to find the maximum input & output size.
def allocate_buffers(engine: trt.ICudaEngine, profile_idx: int | None = None):
    inputs: list[HostDeviceMem] = []
    outputs: list[HostDeviceMem] = []
    bindings: list[int] = []
    stream = cuda_call(cudart.cudaStreamCreate())
    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    for binding in tensor_names:
        # get_tensor_profile_shape returns (min_shape, optimal_shape, max_shape)
        # Pick out the max shape to allocate enough memory for the binding.
        shape = engine.get_tensor_shape(binding) if profile_idx is None else engine.get_tensor_profile_shape(binding, profile_idx)[-1]
        shape_valid = np.all([s >= 0 for s in shape])
        if not shape_valid and profile_idx is None:
            raise ValueError(f"Binding {binding} has dynamic shape, " +\
                "but no profile was specified.")
        size = trt.volume(shape)
        trt_type = engine.get_tensor_dtype(binding)

        # Allocate host and device buffers
        if trt.nptype(trt_type):
            dtype = np.dtype(trt.nptype(trt_type))
            bindingMemory = HostDeviceMem(size, dtype)
        else: # no numpy support: create a byte array instead (BF16, FP8, INT4)
            size = int(size * trt_type.itemsize)
            bindingMemory = HostDeviceMem(size)

        # Append the device buffer to device bindings.
        bindings.append(int(bindingMemory.device))

        # Append to the appropriate list.
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append(bindingMemory)
        else:
            outputs.append(bindingMemory)
    return inputs, outputs, bindings, stream


# Frees the resources allocated in allocate_buffers
def free_buffers(inputs: list[HostDeviceMem], outputs: list[HostDeviceMem], stream: cudart.cudaStream_t):
    for mem in inputs + outputs:
        mem.free()
    cuda_call(cudart.cudaStreamDestroy(stream))


def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray) -> None:
    """
    Copy a numpy array to a device pointer with error checking.
    
    Parameters
    ----------
    device_ptr : int
        The device pointer to copy to.
    host_arr : np.ndarray
        The numpy array to copy.
    
    Raises
    ------
    RuntimeError
        If a CUDA error occurred.
    
    """
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(cudart.cudaMemcpy(device_ptr, host_arr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))


def memcpy_device_to_host(host_arr: np.ndarray, device_ptr: int) -> None:
    """
    Copy a device pointer to a numpy array with error checking.

    Parameters
    ----------
    host_arr : np.ndarray
        The numpy array to copy to.
    device_ptr : int
        The device pointer to copy.

    Raises
    ------
    RuntimeError
        If a CUDA error occurred.
    
    """
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(cudart.cudaMemcpy(host_arr, device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
