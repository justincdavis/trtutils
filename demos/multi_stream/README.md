# Multi-Stream YOLO Demo

A high-performance multi-stream object detection demo showcasing parallel YOLO inference using GPU and DLA engines with TensorRT acceleration.

## Overview

This demo demonstrates concurrent object detection across multiple video streams using YOLO models optimized with TensorRT. It showcases the capability to process multiple video files simultaneously using both GPU and DLA (Deep Learning Accelerator) engines for maximum throughput and performance.

## Features

- **Multi-stream processing** with parallel video streams
- **GPU and DLA engine support** for optimal hardware utilization
- **TensorRT acceleration** for high-performance inference
- **Threaded architecture** with producer-consumer pattern
- **Real-time visualization** (optional) with per-stream display
- **Performance benchmarking** with FPS metrics
- **Queue-based frame management** for smooth processing

## Requirements

- NVIDIA GPU with CUDA support
- TensorRT installed
- NVIDIA Jetson device (for DLA engine support)
- Pre-built TensorRT engines for YOLO models

### Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

## Video Sources

The demo processes 7 MOT17 (Multiple Object Tracking) video sequences:
- `mot17_02.mp4` - Urban street scene
- `mot17_04.mp4` - Crowded public area
- `mot17_05.mp4` - Pedestrian crossing
- `mot17_09.mp4` - Shopping area
- `mot17_10.mp4` - Vehicle traffic
- `mot17_11.mp4` - Busy intersection
- `mot17_13.mp4` - Market scene

You will need to download the MOT17 dataset and assemble the videos.
Or they can be [downloaded here](https://drive.google.com/file/d/1tdVSulYQIlBd86znkGd19jLjmX2N6OcK/view?usp=sharing).

## Engine Requirements

Refer to [YOLOX conversion and build tutorial page](https://trtutils.readthedocs.io/en/latest/tutorials/yolo/yolox.html) on how to convert/build YOLOX models/engines.

The configuration used to get ~190 FPS (GPU + DLA0 + DLA1) on Orin AGX is as follows:

- **GPU:** FP16 precision and all other options default
- **DLA:** INT8 precision, with GPU fallback enabled, COCO17 validation data used for quantization, and all other options default

Both engines used the medium (M) pretrained weights on COCO with an input size of 640x640.

## Quick Start

1. **Clone and navigate to the demo:**
   ```bash
   cd demos/multi_stream
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure engines are available:**
   - Place `yoloxm_gpu.engine` in the `engines/` directory
   - Place `yoloxm_dla.engine` in the `engines/` directory (for DLA support)

4. **Run the demo:**
   ```bash
   python main.py
   ```

## Usage

### Basic Usage (GPU only)
```bash
python main.py
```
Processes all video streams using GPU engine only

### With Display Output
```bash
python main.py --display
```
Shows real-time visualization windows for each stream

### Enable DLA Engines
```bash
python main.py --dla0
```
Adds DLA core 0 for additional parallel processing

```bash
python main.py --dla1
```
Adds DLA core 1 for additional parallel processing

```bash
python main.py --dla0 --dla1 --display
```
Uses all available engines (GPU + both DLA cores)

## How It Works

1. **Engine Loading:** Initializes YOLO models with specified engines (GPU/DLA)
2. **Video Threading:** Spawns producer threads for each video stream
3. **Frame Feeding:** Each video thread feeds frames into a shared input queue
4. **Parallel Processing:** Multiple YOLO threads consume frames and perform inference
5. **Result Collection:** Processed frames with detections are placed in output queue
6. **Visualization:** (Optional) Displays detection results in real-time windows
7. **Performance Metrics:** Reports total processing time and FPS

## Architecture

The demo uses a multi-threaded architecture:

- **Producer Threads:** One per video stream, feeding frames to input queue
- **Consumer Threads:** One per YOLO engine, processing frames from input queue
- **Main Thread:** Handles visualization and coordinates processing
- **Queue Management:** Buffered queues prevent blocking and ensure smooth flow

## Performance

The demo provides comprehensive performance metrics:
- Total frames processed across all streams
- Total processing time
- Overall FPS (frames per second) across all streams

Performance scales with the number of available engines (GPU + DLA cores).

On Orin AGX 64GB when using all available hardware the dataset can be processed 2x faster than only GPU.

Benchmarking run was done with in MAXN powermode with jetson_clocks enabled.

- **GPU Only:** ~95 FPS
- **GPU + 1 DLA:** ~150 FPS
- **GPU + 2 DLA:** ~190 FPS

### Display

Enabling the display results in severe performance degradation since it serializes on CPU resources due to Pythons GIL.
Only enable if you want to verify that models are getting correct outputs during inference.
