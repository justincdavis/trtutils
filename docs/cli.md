# TRTUtils CLI Documentation

This document provides a comprehensive guide to the TRTUtils command-line interface (CLI).

## Overview

TRTUtils provides a command-line interface with several subcommands for working with TensorRT engines and models. The main commands are:

- `benchmark`: Benchmark a TensorRT engine
- `trtexec`: Run trtexec
- `build`: Build a TensorRT engine from an ONNX model
- `can_run_on_dla`: Evaluate if a model can run on a DLA

## Commands

### Benchmark

Benchmark a TensorRT engine to measure its performance metrics.

```bash
trtutils benchmark [options]
```

#### Options

- `--engine, -e`: Path to the engine file (required)
- `--iterations, -i`: Number of iterations to measure over (default: 1000)
- `--warmup_iterations, -wi`: Number of iterations to warmup the model before measuring (default: 100)
- `--jetson, -j`: Use the Jetson-specific benchmarker to record energy and power draw metrics

#### Output

The benchmark command will output:
- Latency metrics (mean, median, min, max) in milliseconds
- Energy consumption metrics (if using Jetson) in Joules
- Power draw metrics (if using Jetson) in Watts

### Build

Build a TensorRT engine from an ONNX model.

```bash
trtutils build [options]
```

#### Options

- `--onnx, -o`: Path to the ONNX model file (required)
- `--output, -out`: Path to save the TensorRT engine file (required)
- `--timing_cache, -tc`: Path to store timing cache data (default: 'timing.cache')
- `--log_level, -ll`: Log level to use if the logger is None (default: WARNING)
- `--workspace, -w`: Workspace size in GB (default: 4.0)
- `--dla_core`: Specify the DLA core (default: engine built for GPU)
- `--calibration_cache, -cc`: Path to store calibration cache data (default: 'calibration.cache')
- `--gpu_fallback`: Allow GPU fallback for unsupported layers when building for DLA
- `--direct_io`: Use direct IO for the engine
- `--prefer_precision_constraints`: Prefer precision constraints
- `--reject_empty_algorithms`: Reject empty algorithms
- `--ignore_timing_mismatch`: Allow different CUDA device timing caches to be used
- `--fp16`: Quantize the engine to FP16 precision
- `--int8`: Quantize the engine to INT8 precision
- `--verbose`: Enable verbose output

> **Note**: The Build API is unstable and experimental with INT8 quantization.

### Can Run on DLA

Evaluate if a model can run on a DLA (Deep Learning Accelerator).

```bash
trtutils can_run_on_dla [options]
```

#### Options

- `--onnx, -o`: Path to the ONNX model file (required)
- `--int8`: Use INT8 precision to assess DLA compatibility
- `--fp16`: Use FP16 precision to assess DLA compatibility
- `--verbose-layers`: Print detailed information about each layer's DLA compatibility
- `--verbose-chunks`: Print detailed information about layer chunks and their device assignments

#### Output

The command will output:
- Whether the model is fully DLA compatible
- The percentage of layers that are compatible with DLA
- If `--verbose-layers` is enabled:
  - Detailed information about each layer including name, type, precision, and metadata
  - DLA compatibility status for each layer
- If `--verbose-chunks` is enabled:
  - Number of layer chunks found
  - For each chunk:
    - Start and end layer indices
    - Number of layers in the chunk
    - Device assignment (DLA or GPU)

### TRTExec

Run trtexec with the provided options.

```bash
trtutils trtexec [options]
```

For detailed information about trtexec options, please refer to the NVIDIA TensorRT documentation.

## Examples

### Benchmarking an Engine

```bash
trtutils benchmark --engine model.engine --iterations 2000 --warmup_iterations 200
```

### Building an Engine from ONNX

```bash
trtutils build --onnx model.onnx --output model.engine --fp16 --workspace 8.0
```

### Checking DLA Compatibility

```bash
# Basic compatibility check
trtutils can_run_on_dla --onnx model.onnx --fp16

# Detailed layer information
trtutils can_run_on_dla --onnx model.onnx --fp16 --verbose-layers

# Detailed chunk information
trtutils can_run_on_dla --onnx model.onnx --fp16 --verbose-chunks

# Full detailed output
trtutils can_run_on_dla --onnx model.onnx --fp16 --verbose-layers --verbose-chunks
```

## Notes

- All paths can be specified as relative or absolute paths
- The CLI automatically sets the log level to INFO when running
- For Jetson-specific features, make sure you're running on a Jetson device
- When using INT8 quantization, ensure you have the appropriate calibration data 
