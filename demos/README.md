# Demos

This directory contains different demos created using trtutils:

## Available Demos

### [axonn](axonn/)
An AxoNN benchmark demo comparing GPU-only, DLA-only, and AxoNN-optimized inference on Jetson hardware with latency, energy, throughput, and power metrics. Supports any model available via `trtutils.download`.

### [multi_stream](multi_stream/)
A high-performance multi-stream object detection demo showcasing parallel YOLO inference using GPU and DLA engines with TensorRT acceleration.

### [yolo_webcam](yolo_webcam/)
A real-time object detection demo using YOLOv10 with TensorRT acceleration for webcam input.

## Running Demos

Each demo has its own README with specific instructions. Generally:

1. Navigate to the demo directory
2. Install requirements: `python3 -m pip install -r requirements.txt`
3. Run the demo: `python3 main.py`

Refer to individual demo READMEs for detailed usage instructions and requirements.
