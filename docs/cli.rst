TRTUtils CLI Documentation
==========================

This document provides a comprehensive guide to the TRTUtils command-line interface (CLI).

Overview
--------

TRTUtils provides a command-line interface with several subcommands for working with TensorRT engines and models. The main commands are:

* ``benchmark``: Benchmark a TensorRT engine
* ``trtexec``: Run trtexec
* ``build``: Build a TensorRT engine from an ONNX model
* ``can_run_on_dla``: Evaluate if a model can run on a DLA
* ``yolo``: Run YOLO inference with TensorRT

Commands
--------

Benchmark
~~~~~~~~~

Benchmark a TensorRT engine to measure its performance metrics.

.. code-block:: console

    python3 -m trtutils benchmark --engine model.engine --iterations 2000 --warmup_iterations 200

Options
^^^^^^^

* ``--engine, -e``: Path to the engine file (required)
* ``--iterations, -i``: Number of iterations to measure over (default: 1000)
* ``--warmup_iterations, -wi``: Number of iterations to warmup the model before measuring (default: 100)
* ``--jetson, -j``: Use the Jetson-specific benchmarker to record energy and power draw metrics

Output
^^^^^^

The benchmark command will output:
* Latency metrics (mean, median, min, max) in milliseconds
* Energy consumption metrics (if using Jetson) in Joules
* Power draw metrics (if using Jetson) in Watts

Build
~~~~~

Build a TensorRT engine from an ONNX model.

.. code-block:: console

    # Basic build with FP16 precision
    python3 -m trtutils build --onnx model.onnx --output model.engine --fp16 --workspace 8.0

    # Build with INT8 quantization using calibration
    python3 -m trtutils build \
        --onnx model.onnx \
        --output model.engine \
        --int8 \
        --calibration_dir ./calibration_images \
        --input_shape 640 640 3 \
        --input_dtype float32 \
        --batch_size 8 \
        --data_order NCHW \
        --resize_method letterbox \
        --input_scale 0.0 1.0

Options
^^^^^^^

* ``--onnx, -o``: Path to the ONNX model file (required)
* ``--output, -out``: Path to save the TensorRT engine file (required)
* ``--timing_cache, -tc``: Path to store timing cache data (default: 'timing.cache')
* ``--log_level, -ll``: Log level to use if the logger is None (default: WARNING)
* ``--workspace, -w``: Workspace size in GB (default: 4.0)
* ``--dla_core``: Specify the DLA core (default: engine built for GPU)
* ``--calibration_cache, -cc``: Path to store calibration cache data (default: 'calibration.cache')
* ``--calibration_dir, -cd``: Directory containing images for INT8 calibration
* ``--input_shape, -is``: Input shape in HWC format (height, width, channels)
* ``--input_dtype, -id``: Input data type (choices: float32, float16, int8)
* ``--batch_size, -bs``: Batch size for calibration (default: 8)
* ``--data_order, -do``: Data ordering expected by the network (choices: NCHW, NHWC, default: NCHW)
* ``--max_images, -mi``: Maximum number of images to use for calibration
* ``--resize_method, -rm``: Method to resize images (choices: letterbox, linear, default: letterbox)
* ``--input_scale, -sc``: Input value range (default: [0.0, 1.0])
* ``--gpu_fallback``: Allow GPU fallback for unsupported layers when building for DLA
* ``--direct_io``: Use direct IO for the engine
* ``--prefer_precision_constraints``: Prefer precision constraints
* ``--reject_empty_algorithms``: Reject empty algorithms
* ``--ignore_timing_mismatch``: Allow different CUDA device timing caches to be used
* ``--fp16``: Quantize the engine to FP16 precision
* ``--int8``: Quantize the engine to INT8 precision
* ``--verbose``: Verbose output from can_run_on_dla

.. note::
   The Build API is unstable and experimental with INT8 quantization.

.. note::
   When using INT8 quantization with calibration, you must provide:
   * ``--calibration_dir``: Directory containing calibration images
   * ``--input_shape``: Expected input shape in HWC format
   * ``--input_dtype``: Expected input data type

Can Run on DLA
~~~~~~~~~~~~~~

Evaluate if a model can run on a DLA (Deep Learning Accelerator).

.. code-block:: console

    # Basic compatibility check
    python3 -m trtutils can_run_on_dla --onnx model.onnx --fp16

    # Detailed layer information
    python3 -m trtutils can_run_on_dla --onnx model.onnx --fp16 --verbose-layers

    # Detailed chunk information
    python3 -m trtutils can_run_on_dla --onnx model.onnx --fp16 --verbose-chunks

    # Full detailed output
    python3 -m trtutils can_run_on_dla --onnx model.onnx --fp16 --verbose-layers --verbose-chunks

Options
^^^^^^^

* ``--onnx, -o``: Path to the ONNX model file (required)
* ``--int8``: Use INT8 precision to assess DLA compatibility
* ``--fp16``: Use FP16 precision to assess DLA compatibility
* ``--verbose-layers``: Print detailed information about each layer's DLA compatibility
* ``--verbose-chunks``: Print detailed information about layer chunks and their device assignments

Output
^^^^^^

The command will output:
* Whether the model is fully DLA compatible
* The percentage of layers that are compatible with DLA
* If ``--verbose-layers`` is enabled:
  * Detailed information about each layer including name, type, precision, and metadata
  * DLA compatibility status for each layer
* If ``--verbose-chunks`` is enabled:
  * Number of layer chunks found
  * For each chunk:
    * Start and end layer indices
    * Number of layers in the chunk
    * Device assignment (DLA or GPU)

TRTExec
~~~~~~~

Run trtexec with the provided options.

.. code-block:: console

    python3 -m trtutils trtexec [options]

For detailed information about trtexec options, please refer to the NVIDIA TensorRT documentation.

YOLO
~~~~

Run YOLO inference with TensorRT.

.. code-block:: console

    # Run inference on a single image
    python3 -m trtutils yolo --engine model.engine --input image.jpg --conf_thres 0.25 --preprocessor cuda

    # Run inference on a video with custom settings
    python3 -m trtutils yolo \
        --engine model.engine \
        --input video.mp4 \
        --conf_thres 0.3 \
        --input_range 0.0 255.0 \
        --preprocessor cpu \
        --resize_method letterbox \
        --warmup \
        --warmup_iterations 20

Options
^^^^^^^

* ``--engine, -e``: Path to the TensorRT engine file (required)
* ``--input, -i``: Path to the input image or video file (required)
* ``--conf_thres, -c``: Confidence threshold for detections (default: 0.1)
* ``--input_range, -r``: Input value range (default: [0.0, 1.0])
* ``--preprocessor, -p``: Preprocessor to use (choices: cpu, cuda, default: cuda)
* ``--resize_method, -rm``: Method to resize images (choices: letterbox, linear, default: letterbox)
* ``--warmup, -w``: Perform warmup iterations
* ``--warmup_iterations, -wi``: Number of warmup iterations (default: 10)
* ``--verbose, -v``: Output additional debugging information

Examples
--------

Benchmarking an Engine
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

    python3 -m trtutils benchmark --engine model.engine --iterations 2000 --warmup_iterations 200

Building an Engine from ONNX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

    # Basic build with FP16 precision
    python3 -m trtutils build --onnx model.onnx --output model.engine --fp16 --workspace 8.0

    # Build with INT8 quantization using calibration
    python3 -m trtutils build \
        --onnx model.onnx \
        --output model.engine \
        --int8 \
        --calibration_dir ./calibration_images \
        --input_shape 640 640 3 \
        --input_dtype float32 \
        --batch_size 8 \
        --data_order NCHW \
        --resize_method letterbox \
        --input_scale 0.0 1.0

Checking DLA Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

    # Basic compatibility check
    python3 -m trtutils can_run_on_dla --onnx model.onnx --fp16

    # Detailed layer information
    python3 -m trtutils can_run_on_dla --onnx model.onnx --fp16 --verbose-layers

    # Detailed chunk information
    python3 -m trtutils can_run_on_dla --onnx model.onnx --fp16 --verbose-chunks

    # Full detailed output
    python3 -m trtutils can_run_on_dla --onnx model.onnx --fp16 --verbose-layers --verbose-chunks

Running YOLO Inference
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

    # Run inference on a single image
    python3 -m trtutils yolo --engine model.engine --input image.jpg --conf_thres 0.25 --preprocessor cuda

    # Run inference on a video with custom settings
    python3 -m trtutils yolo \
        --engine model.engine \
        --input video.mp4 \
        --conf_thres 0.3 \
        --input_range 0.0 255.0 \
        --preprocessor cpu \
        --resize_method letterbox \
        --warmup \
        --warmup_iterations 20

Notes
-----

* All paths can be specified as relative or absolute paths
* The CLI automatically sets the log level to INFO when running
* For Jetson-specific features, make sure you're running on a Jetson device
* When using INT8 quantization, ensure you have the appropriate calibration data 
