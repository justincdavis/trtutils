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
