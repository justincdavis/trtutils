## 0.3.1 (10-29-2024)

### Improved

- Outputs from impls.yolo.YOLO are now all
    Python based types. Improved compatibility with
    JIT compiles (such as numba) and similiar tools
    which need consistent types.

## 0.3.0 (10-25-2024)

### Added

- impls.yolo.ParallelYOLO
    - Allows running multiple YOLO models at once

### Improved

- TRTEngine
    - Now uses async memory copies and execution
    - Now uses pagelocked memory on host

### Removed

- backend submodule
    - Only had pycuda backend which is fully replaced
      with the current cuda-python based engines.

## 0.2.3 (10-17-2024)

### Added

- jetson.benchmark_engine integration with jetsontools > 0.0.3
    Gives latency, energy, and power draw measurements
    for TRTEngines

### Improved

- TRTEngine docstring explanation on threading.

### Fixed

- trtexec.build_engine would build for GPU if DLA
    core was set to 0. Now builds for DLA core 0.

## 0.2.2 (10-17-2024)

### Changed

- TRTEngine now uses execute_async_v2 for inference
    core.create_engine now makes a cudaStream for async.
    Lock for stream creation.

## 0.2.1 (10-16-2024)

### Added

- Locks around TensorRT engine creation and
    CUDA memory allocations. Improves stability
    when allocating engines in parallel.

## 0.2.0 (10-02-2024)

### Added

- benchmark_engine
    Function for benchmarking a TensorRT engine file.
    Measures the host latency of the engine.
- jetson submoudle
    - Currently empty
- impls submodule
- impls.yolo submodule
    YOLO class and associated functions available
    Allows YOLO V7/8/9 exported with EfficientNMS
    to be used and V10 as is.

### Changed

- trtexec.build_from_onnx is now trtexec.build_engine
    With this change also comes some added functionality.
    Can now use .prototxt files to build engines.
    More options available.

## 0.1.2 (10-10-2024)

### Added

- QueuedTRTEngine, QueuedTRTModel, ParallelTRTEngine, ParallelTRTMOdel
    Enable async and parallel execution for engines and models.

## 0.1.1 (07-30-2024)

### Fixed

- During crash, AttributeError could be raised since
    some portions could already be de-allocated.
    Now the error will be caught and removed silently
    if a de-allocation occurs before manual freeing
    of memory.

## 0.1.0 (07-30-2024)

### Changed

- Switched TRTEngine to use cuda-python by default.
    - More stable and robust to differing input/execution dtypes.
    - Official NVIDIA support for Jetson devices using cuda-python.
- Old TRTEngine using PyCUDA can be used by installing trtutils[pycuda]
    and then accessed via. trtutils.backends.PyCudaTRTEngine.

## 0.0.8 (07-21-2024)

### Added

- trtexec submodule
    - Find trtexec command.
    - Ability to run trtexec commands via a Python interface.

## 0.0.3 (02-22-2024)

### Fixed

- Package gets detected as being fully typed.

### Improvements

- Better examples
- Better docs
- Stricter linting and typing

### Added

- Install script for pycuda on Linux
