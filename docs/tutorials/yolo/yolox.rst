.. _tutorials_yolo_x:

YOLOX Tutorial
==============

This tutorial will guide you through using trtutils with YOLOX models.
We will cover:

1. Exporting ONNX weights from YOLOX
2. Converting to end-to-end ONNX
3. Building a TensorRT engine
4. Running inference with the engine
5. Advanced features and optimizations

Exporting ONNX Weights
----------------------

YOLOX requires a two-step process for end-to-end ONNX export. First, export
the basic ONNX weights:

.. code-block:: console

    # Clone the YOLOX repository
    $ git clone https://github.com/Megvii-BaseDetection/YOLOX.git
    $ cd YOLOX

    # Export the ONNX weights
    # VERSION is one of the following: yolox-t, yolox-n, yolox-s, yolox-m
    $ python3 tools/export-onnx.py \
        --output-name ONNX_OUTPUT \
        -n VERSION \
        -c TORCH_WEIGHTS \
        -decode_in_inference

Converting to End-to-End ONNX
-----------------------------

The end-to-end conversion for YOLOX is handled during the engine build process.
You'll need to use the YOLOX-TensorRT repository:

.. code-block:: console

    # Clone the YOLOX-TensorRT repository
    $ git clone https://github.com/justincdavis/YOLOX-TensorRT.git
    $ cd YOLOX-TensorRT

    # Convert the weights to end-to-end TensorRT Engine
    # Adjust parameters according to your needs:
    # - conf-thres: Confidence threshold
    # - iou-thres: IoU threshold for NMS
    # - max-det: Maximum number of detections
    $ python3 export.py \
        -o ONNX_WEIGHT \
        -e ENGINE_OUTPUT \
        --precision 'fp16' \
        --end2end \
        --conf_thres 0.25 \
        --iou_thres 0.5 \
        --max_det 100

    # For Jetson devices with DLA support
    # IMAGE_DIR should contain images for quantization validation
    $ python3 export.py \
        -o ONNX_WEIGHT \
        -e ENGINE_OUTPUT \
        --precision 'fp16' \
        --end2end \
        --conf_thres 0.25 \
        --iou_thres 0.5 \
        --max_det 100 \
        --dlacore 0 \
        --calib_input IMAGE_DIR

    # For better performance on Jetson devices, use INT8 precision
    $ python3 export.py \
        -o ONNX_WEIGHT \
        -e ENGINE_OUTPUT \
        --precision 'int8' \
        --end2end \
        --conf_thres 0.25 \
        --iou_thres 0.5 \
        --max_det 100 \
        --dlacore 0 \
        --calib_input IMAGE_DIR

Running Inference
-----------------

The :py:class:`~trtutils.impls.yolo.YOLO` class provides a high-level interface
for running YOLOX inference. Note that YOLOX requires input images to be in
the range [0, 255]:

.. code-block:: python

    import cv2
    from trtutils.impls.yolo import YOLO, YOLOX

    # Load the YOLOX model with correct input range
    yolo = YOLO("yolox.engine", input_range=(0, 255))

    # OR, use the YOLOX class
    yolo = YOLOX("yolox.engine")

    # Read and process an image
    img = cv2.imread("example.jpg")
    detections = yolo.end2end(img)

    # Print results
    for bbox, confidence, class_id in detections:
        print(f"Class: {class_id}, Confidence: {confidence}")
        print(f"Bounding Box: {bbox}")

Advanced Features
-----------------

Parallel Execution
^^^^^^^^^^^^^^^^^^

You can run multiple YOLOX models in parallel:

.. code-block:: python

    from trtutils.impls.yolo import ParallelYOLO

    # Create a parallel YOLO instance with multiple engines
    yolo = ParallelYOLO(["yolox_1.engine", "yolox_2.engine"])

    # Run inference on multiple images
    images = [cv2.imread(f"image{i}.jpg") for i in range(2)]
    results = yolo.end2end(images)

Benchmarking
^^^^^^^^^^^^

Measure performance with the built-in benchmarking utilities:

.. code-block:: python

    from trtutils import benchmark_engine

    # Run 1000 iterations
    results = benchmark_engine("yolox.engine", iterations=1000)
    print(f"Average latency: {results.latency.mean:.2f}ms")
    print(f"Throughput: {1000/results.latency.mean:.2f} FPS")

    # On Jetson devices, measure power consumption
    from trtutils.jetson import benchmark_engine as jetson_benchmark

    results = jetson_benchmark(
        "yolox.engine",
        iterations=1000,
        tegra_interval=1  # More frequent power measurements
    )
    print(f"Average power draw: {results.power_draw.mean:.2f}W")
    print(f"Total energy used: {results.energy.sum:.2f}J")

Troubleshooting
---------------

Common issues and solutions:

1. **ONNX Export Fails**
   - Ensure you have the correct YOLOX version
   - Check if your PyTorch weights are valid
   - Verify the model architecture matches the export script

2. **Engine Creation Fails**
   - Ensure you have enough GPU memory
   - Check if the ONNX weights are valid

3. **Incorrect Detections**
   - Verify the input image preprocessing matches the training
   - Check if the confidence and IoU thresholds are appropriate
   - Make sure input_range is set to (0, 255)

4. **Performance Issues**
   - Try enabling FP16 precision
   - On Jetson devices, consider using DLA with int8 precision
   - On Jetson devices, ensure MAXN power mode and enable jetson_clocks
