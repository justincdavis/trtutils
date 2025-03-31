0.4.1 (2025-03-04)
------------------

Added
^^^^^
* core.init_cuda
  * Use to start CUDA if not using a TRTEngine and only the core setup.
* benchmark_engines and jetson.benchmark_engines
  * Can benchmark TRTEngines in concurrently running mode
* Example yolo CLI program
  * Currently only support video file input and display.
* General CLI fixes for trtexec
* Experimental non-pagelocked memory addressing for TRTengines
  * Unstable, should be used with caution. Will be refined in the future
  * Does not provide performance improvement, simply for testing speedup
    of pagelocked memory utilization. As such, low-priority
* Basic internal profiling setup for YOLO objects.
  * No current public access, but accessible through: (_pre)(_infer)(_post)_profile attributes
  * Only stores last timestamp tuple
  * No end2end method support yet

Fixed
^^^^^
* yolo.CUDAPreprocessor using the wrong block size during resize call
* Various fixes and extensions to ParallelTRTEngines

0.4.0 (2024-12-05)
------------------

Added
^^^^^
* CUDA-based resize kernels

  * Perform linear or letterbox resizing

* core.create_kernel_args and core.Kernel

0.3.4 (2024-11-12)
------------------

Added
^^^^^
* CUDA-based preprocessing for YOLO:

  * Introduced ``CUDAPreprocessor`` and ``CPUPreprocessor``
  * Additional parameters in YOLO constructor and methods:

    * ``conf_thres``
    * ``extra_nms``, ``agnostic_nms``
    * ``resize_method``, ``preprocessing_unit``

* Runtime CUDA kernel generation with NVRTC:

  * Final transform (transpose from HWC to BCHW) reduced from 50ms to 5ms for 1280x1280, achieving a 10x speedup

Improved
^^^^^^^^
* Multi-threading safety:

  * ``ParallelYOLO`` enforces serial deserialization of engine files
  * ``CUDAProcessor`` now serializes initialization
  * Core CUDA/NVRTC calls use mutexes

0.3.3 (2024-10-31)
------------------

Added
^^^^^
* ``impls.yolo.YOLO``:

  * Added ``input_range`` parameter for specifying the input range
  * ``YOLOX`` uses ``[0:255]``, all others use ``[0:1]``

0.3.2 (2024-10-31)
------------------

Added
^^^^^
* Variations of ``impls.yolo.YOLO``: YOLO7, YOLO8, YOLO9, YOLO10, and YOLOX

Changed
^^^^^^^
* ``impls.yolo.YOLO``:

  * Version inference is now automatic
  * Postprocessing determined from outputs

0.3.1 (2024-10-29)
------------------

Improved
^^^^^^^^
* Outputs from ``impls.yolo.YOLO`` now use standard Python types:

  * Improved compatibility with JIT compilers like ``numba``

0.3.0 (2024-10-25)
------------------

Added
^^^^^
* ``impls.yolo.ParallelYOLO``: Enables running multiple YOLO models simultaneously

Improved
^^^^^^^^
* ``TRTEngine``:

  * Uses async memory copies and execution
  * Implements pagelocked memory on host

Removed
^^^^^^^
* ``backend`` submodule: Deprecated in favor of CUDA Python engines

0.2.3 (2024-10-17)
------------------

Added
^^^^^
* ``jetson.benchmark_engine`` integrated with ``jetsontools > 0.0.3``

Improved
^^^^^^^^
* ``TRTEngine``: Enhanced threading documentation

Fixed
^^^^^
* ``trtexec.build_engine``: Correctly builds for DLA core 0

0.2.2 (2024-10-17)
------------------

Changed
^^^^^^^
* ``TRTEngine``:

  * Uses ``execute_async_v2`` for inference
  * ``core.create_engine`` now creates a ``cudaStream``

0.2.1 (2024-10-16)
------------------

Added
^^^^^
* Locks for TensorRT engine creation and CUDA memory allocation

0.2.0 (2024-10-02)
------------------

Added
^^^^^
* ``benchmark_engine``: Measures engine latency
* Submodules:

  * ``jetson``
  * ``impls``
  * ``impls.yolo``: Supports YOLO variants (V7 to V10)

Changed
^^^^^^^
* ``trtexec.build_from_onnx`` renamed to ``trtexec.build_engine``

0.1.2 (2024-10-10)
------------------

Added
^^^^^
* Async and parallel execution classes:

  * ``QueuedTRTEngine``, ``QueuedTRTModel``
  * ``ParallelTRTEngine``, ``ParallelTRTModel``

0.1.1 (2024-07-30)
------------------

Fixed
^^^^^
* Resolved ``AttributeError`` during deallocation crashes

0.1.0 (2024-07-30)
------------------

Changed
^^^^^^^
* Default ``TRTEngine`` now uses CUDA Python:

  * Improved stability and compatibility
  * Legacy PyCUDA version available via ``trtutils.backends.PyCudaTRTEngine``

0.0.8 (2024-07-21)
------------------

Added
^^^^^
* ``trtexec`` submodule:

  * Locate and run ``trtexec`` commands programmatically

0.0.3 (2024-02-22)
------------------

Fixed
^^^^^
* Correct package detection as fully typed

Improved
^^^^^^^^
* Examples, documentation, and stricter linting/typing

Added
^^^^^
* PyCUDA install script for Linux
