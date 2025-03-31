# trtutils

[![](https://img.shields.io/pypi/pyversions/trtutils.svg)](https://pypi.org/pypi/trtutils/)
![PyPI](https://img.shields.io/pypi/v/trtutils.svg?style=plastic)
[![CodeFactor](https://www.codefactor.io/repository/github/justincdavis/trtutils/badge)](https://www.codefactor.io/repository/github/justincdavis/trtutils)

![MyPy](https://github.com/justincdavis/trtutils/actions/workflows/mypy.yaml/badge.svg?branch=main)
![Ruff](https://github.com/justincdavis/trtutils/actions/workflows/ruff.yaml/badge.svg?branch=main)
![PyPi Build](https://github.com/justincdavis/trtutils/actions/workflows/build-check.yaml/badge.svg?branch=main)
<!-- ![Black](https://github.com/justincdavis/trtutils/actions/workflows/black.yaml/badge.svg?branch=main) -->

A high-level Python interface for TensorRT inference, providing a simple and unified way to run arbitrary TensorRT engines. This library abstracts away the complexity of CUDA memory management, binding management, and engine execution, making it easy to perform inference with any TensorRT engine.

## Features

- Simple, high-level interface for TensorRT inference
- Automatic CUDA memory management
- Support for arbitrary TensorRT engines
- Built-in preprocessing and postprocessing capabilities
- Comprehensive type hints and documentation
- Support for both basic engine execution and end-to-end model inference

## Quick Start

### Basic Engine Usage

The `TRTEngine` class provides a simple interface for running any TensorRT engine:

```python
from trtutils import TRTEngine

# Load your TensorRT engine
engine = TRTEngine("path_to_engine")

# Get input specifications
print(engine.input_shapes)  # Expected input shapes
print(engine.input_dtypes)  # Expected input data types

# Run inference
inputs = read_your_data()
outputs = engine.execute(inputs)
```

### End-to-End Model Inference

The `TRTModel` class allows you to define preprocessing and postprocessing steps along with the engine for complete end-to-end inference:

```python
from trtutils import TRTModel

# Define preprocessing (e.g., image normalization)
def preprocess(inputs):
    return [i / 255 for i in inputs]

# Define postprocessing (e.g., class selection)
def postprocess(outputs):
    return [o[0][0] for o in outputs]

# Create an end-to-end model
model = TRTModel("path_to_engine", preprocess, postprocess)

# Run inference
inputs = read_your_data()
results = model(inputs)
```

## Installation

```bash
pip install trtutils
```

For additional features, you can install optional dependencies:

```bash
# For YOLO model support
pip install "trtutils[yolo]"

# For Jetson device support
pip install "trtutils[jetson]"

# For development
pip install "trtutils[dev]"
```

## Documentation

For detailed documentation, including advanced usage, examples, and API reference, visit our [documentation site](https://trtutils.readthedocs.io/).

## Examples

Check out our [examples directory](examples/) for more detailed usage examples, including:
- Basic engine usage
- End-to-end model inference
- YOLO model implementation
- Benchmarking utilities

## License

This project is licensed under the MIT License - see the LICENSE file for details.
