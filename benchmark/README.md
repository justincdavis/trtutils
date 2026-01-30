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

### Running Benchmarks (`run.py`)

The `run.py` script benchmarks inference performance across frameworks and image sizes.

#### Required Flags

| Flag | Description |
|------|-------------|
| `--device DEVICE` | Name of the device (used for output file naming, e.g., `3080Ti`, `5080`) |
| `--model MODEL` | Model to benchmark. Use `all` to run all supported models |

#### Framework Flags

At least one framework flag is required when running benchmarks:

| Flag | Description |
|------|-------------|
| `--trtutils` | Benchmark using trtutils framework (tests cpu, cuda, trt, and graph modes) |
| `--ultralytics` | Benchmark using ultralytics framework |
| `--sahi` | Benchmark SAHI (Slicing Aided Hyper Inference) comparison |

#### Optional Flags

| Flag | Description |
|------|-------------|
| `--bootstrap` | Download all models for specified sizes upfront, then exit (no benchmarking) |
| `--imgsz SIZE` | Benchmark only a specific image size (e.g., `640`). If omitted, all sizes are tested |
| `--overwrite` | Re-run benchmarks even if data already exists for the model/size |
| `--warmup N` | Number of warmup iterations before timing (default: 100) |
| `--iterations N` | Number of timed iterations for benchmarking (default: 1000) |

#### Examples

```bash
# Bootstrap all models (download only, no benchmarking)
python benchmark/run.py --bootstrap --model all

# Bootstrap a single model
python benchmark/run.py --bootstrap --model yolov10n

# Benchmark single model with trtutils
python benchmark/run.py --device MyDevice --model yolov10n --trtutils

# Benchmark all models with multiple frameworks
python benchmark/run.py --device MyDevice --model all --trtutils --ultralytics

# Benchmark specific image size only
python benchmark/run.py --device MyDevice --model yolov10n --trtutils --imgsz 640

# Re-run and overwrite existing results
python benchmark/run.py --device MyDevice --model yolov10n --trtutils --overwrite

# SAHI comparison benchmarks
python benchmark/run.py --device MyDevice --model yolov10n --sahi
```

Output: Results are saved to `data/models/{device}.json`

## Supported Models

### YOLO Family (Variable Image Sizes)

These models support all image sizes: 160, 320, 480, 640, 800, 960, 1120, 1280

| Family | Models |
|--------|--------|
| YOLOv13 | yolov13n, yolov13s, yolov13m |
| YOLOv12 | yolov12n, yolov12s, yolov12m |
| YOLOv11 | yolov11n, yolov11s, yolov11m |
| YOLOv10 | yolov10n, yolov10s, yolov10m |
| YOLOv9 | yolov9t, yolov9s, yolov9m |
| YOLOv8 | yolov8n, yolov8s, yolov8m |
| YOLOv7 | yolov7t, yolov7m |
| YOLOX | yoloxt, yoloxn, yoloxs, yoloxm |

### DETR Family (Fixed Image Sizes)

These transformer-based models have fixed input size requirements.

| Family | Models | Image Size |
|--------|--------|------------|
| RT-DETR v1 | rtdetrv1_r18, rtdetrv1_r34, rtdetrv1_r50m | 640 |
| RT-DETR v2 | rtdetrv2_r18, rtdetrv2_r34, rtdetrv2_r50m | 640 |
| RT-DETR v3 | rtdetrv3_r18, rtdetrv3_r34, rtdetrv3_r50 | 640 |
| D-FINE | dfine_n, dfine_s, dfine_m | 640 |
| DEIM | deim_dfine_n, deim_dfine_s, deim_dfine_m | 640 |
| DEIM | deim_rtdetrv2_r18, deim_rtdetrv2_r34, deim_rtdetrv2_r50m | 640 |
| DEIMv2 | deimv2_pico, deimv2_n, deimv2_s, deimv2_m | 640 |
| DEIMv2 | deimv2_atto | 320 |
| DEIMv2 | deimv2_femto | 416 |
| RF-DETR | rfdetr_n | 384 |
| RF-DETR | rfdetr_s | 512 |
| RF-DETR | rfdetr_m | 576 |

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

### Generating Plots

The `plot.py` script generates visualizations from benchmark data. It supports three types of plots:

#### `--latency`: Per-Model Latency Bar Charts

Generates bar charts comparing latency across different image sizes (160-1280) and frameworks for each model.

```bash
# All devices
python benchmark/plot.py --latency

# Single device
python benchmark/plot.py --latency --device MyDevice

# Overwrite existing plots
python benchmark/plot.py --latency --device MyDevice --overwrite
```

Output: `plots/{device}/{model}.png` for each model.

#### `--pareto`: Accuracy vs Latency Pareto Frontier

Generates scatter plots showing the trade-off between model accuracy (COCO mAP@50) and latency. Points on the Pareto frontier represent models where no other model is both faster and more accurate.

```bash
# Default framework (trtutils(trt))
python benchmark/plot.py --pareto --device MyDevice

# Specify framework
python benchmark/plot.py --pareto --device MyDevice --framework "trtutils(graph)"
```

Output: `plots/{device}/pareto.png`

#### `--optimizations`: Optimization Speedup Bar Charts

Generates bar charts showing speedup relative to CPU baseline for different optimization configurations (GPU preprocessing, pagelocked memory, unified memory, CUDA graphs).

```bash
# All devices that have data in data/optimizations/
python benchmark/plot.py --optimizations

# Single device
python benchmark/plot.py --optimizations --device MyDevice
```

Output: `plots/{device}/optimizations.png`

### Optimization Benchmarks

Run optimization benchmarks to generate data for `--optimizations` plots (requires `--device`):

```bash
python benchmark/optimizations.py --device MyDevice --model yolov10n
```

Results are saved to `data/optimizations/{device}.json`.

## Notes

- First-time download can take significant time (2-5 min per model)
- Total disk space needed: ~5-10 GB for all models
- Models require license acceptance (handled automatically)
- Internet connection required for downloads
