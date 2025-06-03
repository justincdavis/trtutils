.. _tutorials_yolo:

YOLO Tutorials
==============

This section contains detailed tutorials for using trtutils with different YOLO variants.
Each tutorial covers the complete workflow from exporting ONNX weights to running inference.

Prerequisites
-------------

Before starting with the tutorials, ensure you have:

- TensorRT installed
- CUDA toolkit installed
- A compatible NVIDIA GPU or Jetson device
- Python 3.8 or later
- Git (for cloning repositories)

Optional but recommended:
- Jetson device with DLA support for improved performance
- CUDA-capable GPU with at least 4GB VRAM
- SSD storage for faster model loading

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   YOLOv7 Tutorial <yolov7>
   YOLOv8 Tutorial <yolov8>
   YOLOv9 Tutorial <yolov9>
   YOLOv10 Tutorial <yolov10>
   YOLOX Tutorial <yolox>

Overview
--------

trtutils provides a unified interface for running YOLO models with TensorRT.
The main components are:

1. **YOLO Class**: High-level interface for running inference
   - Handles preprocessing and postprocessing
   - Supports batch processing
   - Provides easy-to-use API for detection results

2. **ParallelYOLO**: Run multiple models in parallel
   - Efficient multi-model inference
   - Automatic resource management
   - Synchronized execution

3. **Benchmarking Tools**: Measure performance and power consumption
   - Latency and throughput measurements
   - Power monitoring on Jetson devices
   - Memory usage tracking

Common Features
---------------

All YOLO variants support:

- End-to-end inference with preprocessing and postprocessing
- Parallel execution of multiple models
- Performance benchmarking
- Power consumption monitoring on Jetson devices
- DLA support for more efficient inference
- Automatic memory management
- Type hints and comprehensive documentation

Model-Specific Notes, Features, and Considerations
--------------------------------------------------

Each YOLO variant has unique requirements:

- **YOLOv7**: Direct end-to-end ONNX export
  - Simple conversion process
  - Poor DLA compatibility

- **YOLOv8**: Requires two-step ONNX conversion
  - Based on Ultralytics framework
  - Poor DLA compatibility

- **YOLOv9**: Manual input shape handling
  - Requires explicit shape specification
  - Poor DLA compatibility

- **YOLOv10**: Requires virtual environment setup
  - Based on Ultralytics framework
  - Removes non-max-suppression at inference time
  - Compatible with YOLOv8 tools
  - Poor DLA compatibility

- **YOLOX**: Requires special input range handling
  - Uses [0, 255] input range
  - Good compatibility with DLA

For detailed instructions specific to each model, see the individual tutorials.

Getting Started
---------------

1. Choose the YOLO variant that best suits your needs
2. Follow the model-specific tutorial for ONNX export
3. Build the TensorRT engine using the provided scripts
4. Use the YOLO class for inference
5. Optimize performance using the advanced features

For best results:
- Start with a small batch size
- Enable FP16 precision when possible
- Use DLA on Jetson devices
- Monitor memory usage
- Benchmark before deployment 
