.. _tutorials_rtdetr:

RT-DETR Tutorials
=================

This section contains detailed tutorials for using trtutils with different RT-DETR variants and related models.
Each tutorial covers the complete workflow from downloading ONNX weights to running inference.

Prerequisites
-------------

Before starting with the tutorials, ensure you have:

- TensorRT installed
- CUDA toolkit installed
- A compatible NVIDIA GPU or Jetson device
- Python 3.8 or later

Optional but recommended:
- Jetson device with DLA support for improved performance
- CUDA-capable GPU with at least 4GB VRAM
- SSD storage for faster model loading

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   RT-DETRv1 Tutorial <rtdetrv1>
   RT-DETRv2 Tutorial <rtdetrv2>
   RT-DETRv3 Tutorial <rtdetrv3>
   D-FINE Tutorial <dfine>
   DEIM Tutorial <deim>
   DEIMv2 Tutorial <deimv2>
   RF-DETR Tutorial <rfdetr>

Overview
--------

trtutils provides a unified interface for running RT-DETR and related models with TensorRT.
The main components are:

1. **RT-DETR Classes**: High-level interfaces for running inference
   - Handles preprocessing and postprocessing
   - Supports batch processing
   - Provides easy-to-use API for detection results

2. **Parallel Processing**: Run multiple models in parallel
   - Efficient multi-model inference
   - Automatic resource management
   - Synchronized execution

3. **Benchmarking Tools**: Measure performance and power consumption
   - Latency and throughput measurements
   - Power monitoring on Jetson devices
   - Memory usage tracking

Common Features
---------------

All RT-DETR variants support:

- End-to-end inference with preprocessing and postprocessing
- Parallel execution of multiple models
- Performance benchmarking
- Power consumption monitoring on Jetson devices
- DLA support for more efficient inference
- Automatic memory management
- Type hints and comprehensive documentation

Model-Specific Notes, Features, and Considerations
--------------------------------------------------

Each RT-DETR variant has unique requirements:

- **RT-DETRv1**: Direct end-to-end ONNX export
  - Simple conversion process
  - Good DLA compatibility
  - Apache-2.0 licensed

- **RT-DETRv2**: Direct end-to-end ONNX export
  - Similar to RT-DETRv1 with improvements
  - Good DLA compatibility
  - Apache-2.0 licensed

- **RT-DETRv3**: PaddlePaddle-based conversion
  - Requires PaddlePaddle to ONNX conversion
  - Limited opset support (max 16)
  - Apache-2.0 licensed

- **D-FINE**: Direct end-to-end ONNX export
  - High performance detection
  - Good DLA compatibility
  - Apache-2.0 licensed

- **DEIM**: Direct end-to-end ONNX export
  - Efficient detection model
  - Good DLA compatibility
  - Apache-2.0 licensed

- **DEIMv2**: Direct end-to-end ONNX export
  - Improved version of DEIM
  - Good DLA compatibility
  - Apache-2.0 licensed

- **RF-DETR**: Direct end-to-end ONNX export
  - Roboflow's DETR implementation
  - Good DLA compatibility
  - Apache-2.0 licensed

For detailed instructions specific to each model, see the individual tutorials.

Getting Started
---------------

1. Choose the RT-DETR variant that best suits your needs
2. Use the trtutils CLI to download and convert models to ONNX
3. Build the TensorRT engine using the provided scripts
4. Use the appropriate model class for inference
5. Optimize performance using the advanced features

For best results:
- Start with a small batch size
- Enable FP16 precision when possible
- Use DLA on Jetson devices
- Monitor memory usage
- Benchmark before deployment
