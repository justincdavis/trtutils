# Benchmark Setup

Benchmarking scripts for trtutils with automatic model downloading.

## Quick Start

### Option 1: Bootstrap All Models (Recommended)

Download all models upfront before benchmarking:

```bash
# Download all models for all sizes
python benchmark/run.py --bootstrap --model all

# Or use make
make benchmark-bootstrap

# Then run benchmarks
python benchmark/run.py --device YourDevice --model yolov10n --trtutils
```

### Option 2: Auto-Download On Demand

Run benchmarks directly - models download automatically as needed:

```bash
python benchmark/run.py --device YourDevice --model yolov10n --trtutils
```

## Usage

### Bootstrap Models
```bash
# All models
python benchmark/run.py --bootstrap --model all

# Single model
python benchmark/run.py --bootstrap --model yolov10n
```

### Run Benchmarks
```bash
# Single model
python benchmark/run.py --device MyDevice --model yolov10n --trtutils

# All models
python benchmark/run.py --device MyDevice --model all --trtutils --ultralytics

# With SAHI
python benchmark/run.py --device MyDevice --model yolov10n --sahi
```

## Supported Models

- **YOLOv13**: yolov13n, yolov13s
- **YOLOv12**: yolov12n, yolov12s, yolov12m
- **YOLOv11**: yolov11n, yolov11s, yolov11m
- **YOLOv10**: yolov10n, yolov10s, yolov10m
- **YOLOv9**: yolov9t, yolov9s, yolov9m
- **YOLOv8**: yolov8n, yolov8s, yolov8m
- **YOLOv7**: yolov7t, yolov7m
- **YOLOX**: yoloxt, yoloxn, yoloxs, yoloxm

Models are tested across image sizes: 160, 320, 480, 640, 800, 960, 1120, 1280

## Implementation

The benchmark system uses `trtutils.download` to automatically download and export models to ONNX format. Models are stored in `data/{model_type}/` directories.

### Files

- **`run.py`**: Main benchmark script comparing trtutils against ultralytics and SAHI frameworks. Supports automatic model downloading with `--bootstrap` and benchmarks across multiple image sizes.
- **`model_utils.py`**: Helper functions for model management including downloading ONNX models and building TensorRT engines.
- **`optimizations.py`**: Benchmarks different Detector optimization combinations (preprocessor type, cuda_graph, pagelocked_mem, unified_mem). Requires `--device`; results are written to `data/optimizations/{device}.json`. Useful for finding the fastest configuration for your hardware.
- **`plot.py`**: Generates latency and Pareto plots from `data/models/`; use `--optimizations` to generate per-device optimization speedup bar charts from `data/optimizations/`.
- **`data/`**: Benchmark data directory:
  - **`data/models/`**: Device benchmark results (one JSON per device, e.g. `3080Ti.json`). Written by `run.py`.
  - **`data/optimizations/`**: Optimization benchmark results (one JSON per device). Written by `optimizations.py`.

### How It Works

1. **Bootstrap mode** (`--bootstrap`): Downloads all specified models upfront
2. **Auto-download**: During benchmarking, missing models are downloaded automatically
3. **Storage**: Models saved as `data/{model_type}/{model}_{size}.onnx`

## Examples

```bash
# Bootstrap only yolov10 models
python benchmark/run.py --bootstrap --model yolov10n

# Run full benchmark suite with pre-downloaded models
make benchmark-bootstrap
python benchmark/run.py --device MyDevice --model all --trtutils --ultralytics

# Quick test with auto-download
python benchmark/run.py --device MyDevice --model yolov8n --trtutils
```

### Optimization benchmarks and plots

Run optimization benchmarks (requires `--device`); results are saved to `data/optimizations/{device}.json`:

```bash
python benchmark/optimizations.py --device MyDevice --model yolov10n
```

Generate per-device optimization speedup bar charts (CPU baseline = 1.0x, then GPU preproc, pagelocked, unified, CUDA graph, etc.):

```bash
# All devices that have data in data/optimizations/
python benchmark/plot.py --optimizations

# Single device
python benchmark/plot.py --optimizations --device MyDevice
```

Plots are written to `plots/{device}/optimizations.png`.

## Notes

- First-time download can take significant time (2-5 min per model)
- Total disk space needed: ~5-10 GB for all models
- Models require license acceptance (handled automatically)
- Internet connection required for downloads
