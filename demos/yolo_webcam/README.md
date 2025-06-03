# YOLO Webcam Demo

A real-time object detection demo using YOLOv10 with TensorRT acceleration for webcam input.

## Overview

This demo demonstrates real-time object detection using a YOLOv10 nano model optimized with TensorRT. It captures video from your webcam and displays bounding boxes around detected objects in real-time.

## Features

- **Real-time object detection** with YOLOv10 nano model
- **TensorRT acceleration** for optimal performance
- **Webcam integration** with live video feed
- **Interactive display** with frame counter
- **Automatic engine building** from ONNX model

## Requirements

- NVIDIA GPU with CUDA support
- TensorRT installed
- Webcam or video capture device

### Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

## Quick Start

1. **Clone and navigate to the demo:**
   ```bash
   cd demos/yolo_webcam
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the demo:**
   ```bash
   python main.py
   ```

4. **Exit:** Press 'q' or close the window to stop the demo

## Usage

### Basic Usage
```bash
python main.py
```
Uses default webcam (device 0)

### Specify Video Source
```bash
python main.py --source 1
```
Use different camera device (e.g., device 1)

## Model Information

- **Model:** YOLOv10 Nano (`yolov10n.onnx`)
- **Size:** ~9MB ONNX, ~8MB TensorRT engine
- **Optimization:** FP16 precision for faster inference
- **Preprocessing:** TensorRT-based with letterbox resizing

## How It Works

1. **Engine Building:** On first run, automatically converts ONNX model to optimized TensorRT engine
2. **Model Loading:** Loads YOLOv10 with TensorRT backend and performs warmup
3. **Video Capture:** Captures frames from specified video source
4. **Inference:** Runs end-to-end object detection on each frame
5. **Visualization:** Draws bounding boxes and displays results in real-time
