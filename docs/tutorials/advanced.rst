.. _tutorials_advanced:

Advanced Usage Tutorial
=======================

This tutorial covers advanced usage of trtutils, including lower-level interfaces
and CUDA operations.

trtutils provides some lower-level interfaces which are used inside of 
:py:class:`~trtutils.TRTEngine`. These involve creating TensorRT engines 
allocating memory with CUDA.

These lower-level utilities can be found inside of the :py:mod:`~trtutils.core` 
submodule. All utilties included are:

1. CUDA context helpers

2. CUDA stream helpers

3. CUDA memory allocation and transfer functions

4. NVRTC to compile CUDA kernels

5. TensorRT engine deserialization

6. Binding abstraction for managed memory


Bindings
^^^^^^^^

:py:class:`~trtutils.core.Binding` manages CUDA-allocated memory. 
You can create bindings directly if you allocate memory manually with :py:func:`~trtutils.core.cuda_malloc` 
or :py:func:`~trtutils.core.allocate_pinned_memory`. CudaMalloc allocates memory
directly on the GPU and allocate_pinned_memory allocates page-locked memory to share between 
CUDA and the CPU. Pagelocked memory can enable large speedup on some systems. 
You can also create a binding with :py:func:`~trtutils.core.create_binding` which 
automatically allocates memory based on a given Numpy array.

Example of binding allocation with create_binding:

.. code-block:: python

    import numpy as np
    from trtutils.core import create_binding

    # float16 image
    arr = np.random.default_rng().integers(0, 255, (480, 640, 3), dtype=np.float16)

    # get a binding to represent the array
    binding = create_binding(arr)

    # allocate with pagelocked memory
    pl_binding = create_binding(arr, pagelocked_mem=True)

TensorRT Engine deserialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can deserialize a TensorRT engine using the :py:func:`~trtutils.core.create_engine` 
function. This function also allocates an execution context, a tensorrt logger, and a CUDA
stream which can be used to execute the engine.

.. code-block:: python

    from trtutils.core import create_engine

    # given a path to a compiled TensorRT engine, deserialize
    # tensorrt.ICudaEngine, tensorrt.IExecutionContext, tensorrt.ILogger, cuda.cudaStream_t 
    engine, context, logger, stream = create_engine("engine.engine")

CUDA Kernel Compilation
^^^^^^^^^^^^^^^^^^^^^^^

It may be necessary to define preprocessing or postprocessing operations
for your TensorRT engine with a CUDA kernel if the CPU is not fast enough.
trtutils provides a small wrapper around NVRTC (NVIDIA real-time compiler)
which compiles a CUDA source file at runtime.

:class:`~trtutils.core.Kernel` reads CUDA source from a ``.cu`` file.
``create_args`` packs device pointers and scalar arguments, and ``call``
launches the kernel on a stream. See ``examples/kernel.py`` for a complete
program. A short version:

.. code-block:: python

    import tempfile
    from pathlib import Path

    import numpy as np

    from trtutils.core import (
        Kernel,
        create_binding,
        create_stream,
        memcpy_device_to_host_async,
        memcpy_host_to_device_async,
        stream_synchronize,
    )

    kernel_code = """\
    extern "C" __global__
    void scaleSwapTranspose(
        const unsigned char* __restrict__ inputArr,
        float* outputArr,
        const float scale,
        const float offset,
        const int height,
        const int width
    ) {
        const int tx = blockIdx.x * blockDim.x + threadIdx.x;
        const int ty = blockIdx.y * blockDim.y + threadIdx.y;
        const int tz = blockIdx.z * blockDim.z + threadIdx.z;
        if (tx < height && ty < width && tz < 3) {
            const int inputIdx = (tx * width * 3) + (ty * 3) + tz;
            const float val = static_cast<float>(inputArr[inputIdx]);
            const float scaledVal = val * scale + offset;
            const int dstChannel = 2 - tz;
            const int outputIdx = (dstChannel * height * width) + (tx * width) + ty;
            outputArr[outputIdx] = scaledVal;
        }
    }
    """

    with tempfile.NamedTemporaryFile("w", suffix=".cu", delete=False) as f:
        f.write(kernel_code)
        kernel_path = Path(f.name)

    kernel = Kernel(kernel_path, "scaleSwapTranspose")
    kernel_path.unlink()

    height, width = 640, 640
    scale, offset = 1.0 / 255.0, 0.0
    input_arr = np.zeros((height, width, 3), dtype=np.uint8)
    output_arr = np.zeros((1, 3, height, width), dtype=np.float32)
    input_binding = create_binding(input_arr)
    output_binding = create_binding(output_arr)

    args = kernel.create_args(
        input_binding.allocation,
        output_binding.allocation,
        scale,
        offset,
        height,
        width,
    )

    stream = create_stream()
    memcpy_host_to_device_async(input_binding.allocation, input_arr, stream)
    # 16x16 threads per block, one block row per channel
    threads = (16, 16, 1)
    blocks = ((height + 15) // 16, (width + 15) // 16, 3)
    kernel.call(blocks, threads, stream, args)
    memcpy_device_to_host_async(output_binding.host_allocation, output_binding.allocation, stream)
    stream_synchronize(stream)
    print(output_binding.host_allocation.shape)
    kernel.free()
