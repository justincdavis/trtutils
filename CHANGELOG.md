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
