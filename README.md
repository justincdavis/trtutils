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

## Installation

```bash
pip install trtutils
```

For additional features, you can install optional dependencies:

```bash
# For JIT compiler
pip install "trtutils[jit]"

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

## Performance

| Device            | YOLOv8m                                                                 | YOLOv8n                                                                 |
|-------------------|-------------------------------------------------------------------------|-------------------------------------------------------------------------|
| OrinAGX-64GB      | ![OrinAGX-64GB YOLOv8m](benchmark/plots/OrinAGX-64GB/yolov8m.png)       | ![OrinAGX-64GB YOLOv8n](benchmark/plots/OrinAGX-64GB/yolov8n.png)       |
| OrinAGX-32GB      | ![OrinAGX-32GB YOLOv8m](benchmark/plots/OrinAGX-32GB/yolov8m.png)       | ![OrinAGX-32GB YOLOv8n](benchmark/plots/OrinAGX-32GB/yolov8n.png)       |
| OrinNX-16GB       | ![OrinNX-16GB YOLOv8m](benchmark/plots/OrinNX-16GB/yolov8m.png)         | ![OrinNX-16GB YOLOv8n](benchmark/plots/OrinNX-16GB/yolov8n.png)        |
| OrinNano-8GB      | ![OrinNano-8GB YOLOv8m](benchmark/plots/OrinNano-8GB/yolov8m.png)       | ![OrinNano-8GB YOLOv8n](benchmark/plots/OrinNano-8GB/yolov8n.png)       |
| XavierNX-8GB      | ![XavierNX-8GB YOLOv8m](benchmark/plots/XavierNX-8GB/yolov8m.png)       | ![XavierNX-8GB YOLOv8n](benchmark/plots/XavierNX-8GB/yolov8n.png)       |
| 3080Ti            | ![3080Ti YOLOv8m](benchmark/plots/3080Ti/yolov8m.png)                   | ![3080Ti YOLOv8n](benchmark/plots/3080Ti/yolov8n.png)                   |
| TitanRTX          | ![TitanRTX YOLOv8m](benchmark/plots/TitanRTX/yolov8m.png)               | ![TitanRTX YOLOv8n](benchmark/plots/TitanRTX/yolov8n.png)               |

## License

This project is licensed under the MIT License - see the LICENSE file for details.
