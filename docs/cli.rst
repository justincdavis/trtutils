TRTUtils CLI Documentation
==========================

This document provides a comprehensive guide to the TRTUtils command-line interface (CLI).

Overview
--------

TRTUtils provides a command-line interface with several subcommands for working with TensorRT engines and models. The main commands are:

* ``benchmark``: Benchmark a TensorRT engine
* ``build``: Build a TensorRT engine from an ONNX model
* ``build_dla``: Build a TensorRT engine with mixed GPU/DLA layers and precision automatically
* ``can_run_on_dla``: Evaluate if a model can run on a DLA and specific layer/chunk compatibility.
* ``classify``: Run image classification on an image
* ``detect``: Run object detection on an image or video
* ``download``: Download a model from remote source and convert to ONNX
* ``inspect``: Inspect a TensorRT engine
* ``trtexec``: Run trtexec with the provided options

Global Options
--------------

These options are available for most commands:

* ``--dla_core``: DLA core to assign DLA layers of the engine to (default: None)
* ``--log_level``: Set the log level (choices: DEBUG, INFO, WARNING, ERROR, CRITICAL; default: INFO)
* ``--verbose``: Enable verbose output

Commands
--------

Benchmark
~~~~~~~~~

Benchmark a TensorRT engine to measure its performance metrics.

.. code-block:: console

    # Basic benchmarking
    python3 -m trtutils benchmark --engine model.engine --iterations 2000

    # Jetson benchmarking with energy/power metrics
    python3 -m trtutils benchmark --engine model.engine --jetson --tegra_interval 1

    # With warmup
    python3 -m trtutils benchmark --engine model.engine --warmup --warmup_iterations 200

Options
^^^^^^^

* ``--engine, -e``: Path to the engine file (required)
* ``--iterations, -i``: Number of iterations to measure over (default: 1000)
* ``--jetson, -j``: Use the Jetson-specific benchmarker to record energy and power draw metrics
* ``--tegra_interval``: Milliseconds between each tegrastats sampling for Jetson benchmarking (default: 5)
* ``--warmup``: Perform warmup iterations
* ``--warmup_iterations, -wi``: Number of warmup iterations (default: 10)

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

    # Basic FP32 build
    python3 -m trtutils build --onnx model.onnx --output model.engine

    # FP16 build
    python3 -m trtutils build --onnx model.onnx --output model.engine --fp16

    # INT8 build with calibration
    python3 -m trtutils build \
        --onnx model.onnx \
        --output model.engine \
        --int8 \
        --calibration_dir ./calibration_images \
        --input_shape 640 640 3 \
        --input_dtype float32

Options
^^^^^^^

**Required:**

* ``--onnx, -o``: Path to the ONNX model file
* ``--output, -out``: Path to save the TensorRT engine file

**Build Configuration:**

* ``--device, -d``: Device to use for the engine (choices: gpu, dla; default: gpu)
* ``--workspace, -w``: Workspace size in GB (default: 4.0)
* ``--fp16``: Quantize the engine to FP16 precision
* ``--int8``: Quantize the engine to INT8 precision
* ``--gpu_fallback``: Allow GPU fallback for unsupported layers when building for DLA

**Caching and Optimization:**

* ``--timing_cache, -tc``: Path to store timing cache data
* ``--calibration_cache, -cc``: Path to store calibration cache data
* ``--cache``: Cache the engine in the trtutils engine cache
* ``--direct_io``: Use direct IO for the engine
* ``--prefer_precision_constraints``: Prefer precision constraints
* ``--reject_empty_algorithms``: Reject empty algorithms
* ``--ignore_timing_mismatch``: Allow different CUDA device timing caches to be used

**Calibration (for INT8):**

* ``--calibration_dir, -cd``: Directory containing images for INT8 calibration
* ``--input_shape, -is``: Input shape in HWC format (height, width, channels)
* ``--input_dtype, -id``: Input data type (choices: float32, float16, int8)
* ``--batch_size, -bs``: Batch size for calibration (default: 8)
* ``--data_order, -do``: Data ordering expected by the network (choices: NCHW, NHWC; default: NCHW)
* ``--max_images, -mi``: Maximum number of images to use for calibration
* ``--resize_method, -rm``: Method to resize images (choices: letterbox, linear; default: letterbox)
* ``--input_scale, -sc``: Input value range (default: [0.0, 1.0])

.. note::
   When using INT8 quantization with calibration, you must provide:
   * ``--calibration_dir``: Directory containing calibration images
   * ``--input_shape``: Expected input shape in HWC format
   * ``--input_dtype``: Expected input data type

Build DLA
~~~~~~~~~

Build a TensorRT engine for DLA, supporting mixed GPU/DLA layers and precision.

.. code-block:: console

    python3 -m trtutils build_dla \
        --onnx model.onnx \
        --output model.engine \
        --dla_core 0 \
        --max_chunks 1 \
        --min_layers 20 \
        --calibration_dir ./calibration_images \
        --input_shape 640 640 3 \
        --input_dtype float32 \
        --batch_size 8 \
        --data_order NCHW \
        --resize_method letterbox \
        --input_scale 0.0 1.0

Options
^^^^^^^

**Required:**

* ``--onnx, -o``: Path to the ONNX model file
* ``--output, -out``: Path to save the TensorRT engine file
* ``--calibration_dir, -cd``: Directory containing images for calibration (required for DLA)
* ``--input_shape, -is``: Input shape in HWC format (required for DLA)
* ``--input_dtype, -id``: Input data type (required for DLA)

**DLA Configuration:**

* ``--max_chunks``: Maximum number of DLA chunks to assign (default: 1)
* ``--min_layers``: Minimum number of layers in a chunk to be assigned to DLA (default: 20)

**Other options:** Same as the ``build`` command for calibration, caching, and optimization settings.

Can Run on DLA
~~~~~~~~~~~~~~

Evaluate if a model can run on a DLA (Deep Learning Accelerator).

.. code-block:: console

    # Basic compatibility check
    python3 -m trtutils can_run_on_dla --onnx model.onnx

    # Detailed layer information
    python3 -m trtutils can_run_on_dla --onnx model.onnx --verbose_layers

    # Detailed chunk information
    python3 -m trtutils can_run_on_dla --onnx model.onnx --verbose_chunks

    # Full detailed output
    python3 -m trtutils can_run_on_dla --onnx model.onnx --verbose_layers --verbose_chunks

Options
^^^^^^^

* ``--onnx, -o``: Path to the ONNX model file (required)
* ``--verbose_layers``: Print detailed information about each layer's DLA compatibility
* ``--verbose_chunks``: Print detailed information about layer chunks and their device assignments

Output
^^^^^^

The command will output:

* Whether the model is fully DLA compatible
* The percentage of layers that are compatible with DLA
* If ``--verbose_layers`` is enabled:

  * Detailed information about each layer including name, type, precision, and metadata
  * DLA compatibility status for each layer

* If ``--verbose_chunks`` is enabled:

  * Number of layer chunks found
  * For each chunk:

    * Start and end layer indices
    * Number of layers in the chunk
    * Device assignment (DLA or GPU)

Classify
~~~~~~~~

Run image classification on an image with comprehensive configuration options.

.. code-block:: console

    # Basic image classification
    python3 -m trtutils classify --engine model.engine --input image.jpg --show

    # With warmup and custom configuration
    python3 -m trtutils classify \
        --engine model.engine \
        --input image.jpg \
        --warmup \
        --warmup_iterations 20 \
        --preprocessor cuda \
        --input_range 0.0 1.0 \
        --pagelocked_mem \
        --verbose

Options
^^^^^^^

**Required:**

* ``--engine, -e``: Path to the TensorRT engine file
* ``--input, -i``: Path to the input image file

**Preprocessing:**

* ``--input_range, -r``: Input value range (default: [0.0, 1.0])
* ``--preprocessor, -p``: Preprocessor to use (choices: cpu, cuda, trt; default: trt)

**Memory and Performance:**

* ``--warmup``: Perform warmup iterations
* ``--warmup_iterations, -wi``: Number of warmup iterations (default: 10)
* ``--pagelocked_mem``: Use pagelocked memory for CUDA operations
* ``--unified_mem``: Use unified memory for CUDA operations
* ``--no_warn``: Suppress warnings from TensorRT

**Display:**

* ``--show``: Show the classification results (opens display window)

Output
^^^^^^

The command will output:
* Classification result (class index and confidence score)
* Timing information for each stage:

  * Preprocessing time in milliseconds
  * Inference time in milliseconds
  * Postprocessing time in milliseconds
  * Classification parsing time in milliseconds

Detect
~~~~~~

Run object detection on an image or video with comprehensive configuration options.

.. code-block:: console

    # Basic image inference
    python3 -m trtutils detect --engine model.engine --input image.jpg --show

    # Video inference with custom thresholds
    python3 -m trtutils detect \
        --engine model.engine \
        --input video.mp4 \
        --conf_thres 0.25 \
        --nms_iou_thres 0.45 \
        --preprocessor cuda \
        --show

    # Advanced configuration
    python3 -m trtutils detect \
        --engine model.engine \
        --input image.jpg \
        --warmup \
        --warmup_iterations 20 \
        --pagelocked_mem \
        --extra_nms \
        --agnostic_nms \
        --verbose

Options
^^^^^^^

**Required:**

* ``--engine, -e``: Path to the TensorRT engine file
* ``--input, -i``: Path to the input image or video file

**Detection Configuration:**

* ``--conf_thres, -c``: Confidence threshold for detections (default: 0.1)
* ``--nms_iou_thres``: NMS IOU threshold for detections (default: 0.5)
* ``--extra_nms``: Perform additional CPU-side NMS
* ``--agnostic_nms``: Perform class-agnostic NMS

**Preprocessing:**

* ``--input_range, -r``: Input value range (default: [0.0, 1.0])
* ``--preprocessor, -p``: Preprocessor to use (choices: cpu, cuda, trt; default: trt)
* ``--resize_method, -rm``: Method to resize images (choices: letterbox, linear; default: letterbox)

**Memory and Performance:**

* ``--warmup``: Perform warmup iterations
* ``--warmup_iterations, -wi``: Number of warmup iterations (default: 10)
* ``--pagelocked_mem``: Use pagelocked memory for CUDA operations
* ``--unified_mem``: Use unified memory for CUDA operations
* ``--no_warn``: Suppress warnings from TensorRT

**Display:**

* ``--show``: Show the detections (opens display window)

Output
^^^^^^

The command will output timing information for each stage:
* Preprocessing time in milliseconds
* Inference time in milliseconds
* Postprocessing time in milliseconds
* Detection parsing time in milliseconds

Download
~~~~~~~~

Download a model from remote source and convert to ONNX format. This command supports automatic downloading and conversion of various YOLO and DETR models.

.. code-block:: console

    # Download YOLOv8n
    python3 -m trtutils download --model yolov8n --output yolov8n.onnx --accept

    # Download YOLOv11m with custom settings
    python3 -m trtutils download --model yolov11m --output yolov11m.onnx --imgsz 640 --opset 17 --accept

    # Download YOLOX small model
    python3 -m trtutils download --model yoloxs --output yoloxs.onnx --imgsz 640 --opset 17 --accept

    # Download RT-DETRv1 model
    python3 -m trtutils download --model rtdetrv1_r18vd --output rtdetrv1.onnx --accept

Options
^^^^^^^

**Required:**

* ``--model``: Name of the model to download. See :ref:`Supported Models <models>` for available models
* ``--output``: Path to save the ONNX model file

**Optional:**

* ``--opset``: ONNX opset version to use (default: 17)
* ``--imgsz``: Image size to use for the model (default: 640)
* ``--accept``: Accept the license terms for the model. If not provided, you will be prompted.

Supported Models
^^^^^^^^^^^^^^^^

**YOLO Models:**

* YOLOv7: All variants with pretrained weights
* YOLOv8: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x (via Ultralytics)
* YOLOv9: All variants with pretrained weights
* YOLOv10: All variants with pretrained weights
* YOLOv11: yolov11n, yolov11s, yolov11m, yolov11l, yolov11x (via Ultralytics)
* YOLOv12: All variants with pretrained weights
* YOLOv13: yolov13n, yolov13s, yolov13l, yolov13x
* YOLOX: yoloxn, yoloxt, yoloxs, yoloxm, yoloxl, yoloxx, yolox_darknet

**DETR Models:**

* RT-DETRv1, RT-DETRv2, RT-DETRv3: Multiple configurations
* D-FINE: Multiple configurations
* DEIM: Multiple configurations
* DEIMv2: deimv2_atto, deimv2_femto, deimv2_pico, deimv2_n, deimv2_s, deimv2_m, deimv2_l, deimv2_x
* RF-DETR: Multiple configurations

Notes
^^^^^

* The download process will create a temporary virtual environment to handle dependencies
* Some models may have license restrictions (GPL-3.0, AGPL-3.0, Apache-2.0)
* RT-DETRv3 and RF-DETR do not support custom input sizes
* DEIMv2 does not support custom input sizes

Inspect
~~~~~~~

Inspect a TensorRT engine for metadata and IO information.

.. code-block:: console

    # Basic inspection
    python3 -m trtutils inspect --engine model.engine

    # Verbose inspection
    python3 -m trtutils inspect --engine model.engine --verbose

Options
^^^^^^^

* ``--engine, -e``: Path to the engine file (required)

Output
^^^^^^

The inspect command will output:
* Engine size in MB
* Max batch size
* Input and output tensor names, shapes, data types, and formats

TRTExec
~~~~~~~

Run trtexec with the provided options. This command passes all arguments directly to the native trtexec binary.

.. code-block:: console

    # Build engine with trtexec
    python3 -m trtutils trtexec --onnx=model.onnx --saveEngine=model.engine --fp16

    # Benchmark with trtexec
    python3 -m trtutils trtexec --loadEngine=model.engine --iterations=1000

Options
^^^^^^^

All standard trtexec options are supported. Refer to the TensorRT documentation for complete trtexec usage.

Parent Parser Organization
--------------------------

The CLI is organized using parent parsers to avoid duplication:

* **global_parser**: Common options like ``--dla_core``, ``--log_level``, ``--verbose``
* **build_common_parser**: Build-related options like ``--timing_cache``, ``--workspace``, optimization flags
* **calibration_parser**: Calibration options for INT8 quantization
* **warmup_parser**: Warmup-related options like ``--warmup``, ``--warmup_iterations``
* **memory_parser**: Memory management options like ``--pagelocked_mem``, ``--unified_mem``, ``--no_warn``

This organization ensures consistency across commands and reduces code duplication while maintaining comprehensive parameter coverage.
