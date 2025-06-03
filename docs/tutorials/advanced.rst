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
or :py:func:`~trtutils.core.allocate_pagelocked`. CudaMalloc allocates memory 
directly on the GPU and allocate_pagelocked allocates page-locked memory to share between 
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

It may be nessecary to define preprocessing or postprocessing operations
for your TensorRT engine with a CUDA kernel if the CPU is not fast enough.
trtutils provides a small wrapper around NVRTC (NVIDIA real-time compiler) 
which allows you to define CUDA kernels as Python strings and compile at 
runtime.

An example of compiling a kernel:

.. code-block:: python

    # kernel which handles preprocessing for a YOLO model
    # after the image has been resize to the models input size
    # this assumes the model takes RGB and image is BGR (OpenCV)
    KERNEL_CODE = """\
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

    from trtutils.core import Kernel

    # compile and load kernel
    kernel = Kernel(KERNEL_CODE, "scaleSwapTranspose")

    # to run the kernel need input and output CUDA data
    import numpy as np
    from trtutils.core import create_binding

    input_arr = np.zeros((640, 640, 3), dtype=np.uint8)
    output_arr = np.zeros((1, 3, 640, 640), dtype=np.float32)

    input_binding = create_binding(input_arr)
    output_binding = create_binding(output_arr)

    # create some args for the kernel
    # the args is a pointer to an array of pointers
    # a new arg array has to be created for each call to cuLaunchKernel
    input_arg: np.ndarray = np.array(
        [input_binding.allocation],
        dtype=np.uint64,
    )
    output_arg: np.ndarray = np.array(
        [output_binding.allocation],
        dtype=np.uint64,
    )
    # assume no scale and offset
    args = kernel.create_args(
        input_binding.allocation,
        output_binding.allocation,
        height,
        width,
        scale,
        offset,
    )

    # launch the kernel
    from trtutils.core import create_stream, stream_synchronize, memcpy_host_to_device_async, memcpy_device_to_host_async

    stream = create_stream()
    memcpy_host_to_device_async(
        input_binding.allocation,
        input_arr,
        stream,
    )
    kernel.call((32, 32, 1), (32, 32, 1), stream, args)
    memcpy_device_to_host_async(
        output_binding.host_allocation,
        output_binding.allocation,
        stream,
    )
    stream_synchronize(stream)

    # print the completed output shape
    print(output_binding.host_allocation.shape)
