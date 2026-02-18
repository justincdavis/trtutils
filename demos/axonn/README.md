# AxoNN Benchmark

Benchmarks any model supported by `trtutils.download` on Jetson hardware, comparing three engine configurations: GPU-only (FP16), DLA-only (INT8 with GPU fallback), and AxoNN-optimized (automatic GPU/DLA layer mapping).

## Overview

[AxoNN](../../src/trtutils/research/axonn/) optimizes neural network inference on Jetson devices by finding optimal GPU/DLA layer mappings that minimize execution time within an energy budget. This demo downloads a model, builds three TensorRT engines, and benchmarks them to produce a comparison of latency, energy, throughput, and power metrics.

## Requirements

- NVIDIA Jetson device (Orin AGX, Orin NX, etc.)
- TensorRT installed
- [uv](https://docs.astral.sh/uv/) >= 0.9.0 (for model downloading)

### Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

## Quick Start

1. **Navigate to the demo:**
   ```bash
   cd demos/axonn
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the demo:**
   ```bash
   python main.py
   ```

The demo will automatically download the ResNet34 ONNX model (default), build all three engines, benchmark them, and print a comparison table.

## Usage

### Basic Usage (ResNet34)
```bash
python main.py
```

### Different Model
```bash
python main.py --model resnet18
python main.py --model mobilenet_v2
python main.py --model squeezenet1_1
```

### Custom AxoNN Parameters
```bash
python main.py --energy-ratio 0.6 --max-transitions 5
```

### Quick Test
```bash
python main.py --iterations 100 --verbose
```

### Force Rebuild
```bash
python main.py --rebuild
```

## CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model` | str | resnet34 | Model name (any from `trtutils.download.get_supported_models()`) |
| `--imgsz` | int | auto | Image size (default: model's default) |
| `--energy-ratio` | float | 0.8 | AxoNN energy target ratio (0.0-1.0) |
| `--max-transitions` | int | 3 | Max GPU/DLA boundary crossings |
| `--iterations` | int | 1000 | Benchmark iterations |
| `--warmup-iterations` | int | 50 | Warmup iterations |
| `--profile-iterations` | int | 1000 | AxoNN profiling iterations |
| `--dla-core` | int | 0 | DLA core (0 or 1) |
| `--rebuild` | flag | - | Force rebuild all engines |
| `--verbose` | flag | - | Verbose output |

## How It Works

1. **ONNX Download:** Downloads and exports the model to ONNX format via `trtutils.download`
2. **Input Detection:** Reads input tensor name and shape from the ONNX file
3. **GPU Engine Build:** Builds a standard FP16 TensorRT engine using only the GPU
4. **DLA Engine Build:** Builds an INT8 engine targeting DLA with GPU fallback for unsupported layers
5. **AxoNN Engine Build:** Profiles each layer on both GPU and DLA, then uses AxoNN's optimization algorithm to find the best GPU/DLA mapping within the energy budget
6. **Benchmarking:** Runs each engine for the specified iterations while measuring latency and power via tegrastats
7. **Results:** Computes and displays metrics with percentage comparisons
