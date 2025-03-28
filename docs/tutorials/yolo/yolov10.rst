.. _tutorials_yolo_v10:

YOLOv10 Tutorial
================

This tutorial will guide you through using trtutils with YOLOv10 models.
We will cover:

1. Exporting ONNX weights from YOLOv10
2. Converting to end-to-end ONNX
3. Building a TensorRT engine
4. Running inference with the engine
5. Advanced features and optimizations

Exporting ONNX Weights
----------------------

YOLOv10 is built on top of Ultralytics' framework and requires a virtual environment
to avoid conflicts with other packages. Here's how to export the ONNX weights:

.. code-block:: console

    # Clone the YOLOv10 repository
    $ git clone https://github.com/THU-MIG/yolov10.git
    $ cd yolov10

    # Create and activate a virtual environment
    $ python3 -m venv yolov10
    $ source yolov10/bin/activate

    # Install dependencies
    $ pip install -r requirements.txt

    # Export the ONNX weights
    # Adjust parameters according to your needs:
    # - opset: ONNX opset version
    # - simplify: Enable ONNX simplification
    # - imgsz: Input image size
    $ yolo export \
        weights=PATH_TO_WEIGHTS \
        format=onnx \
        opset=13 \
        simplify \
        imgsz=640

    # Deactivate the virtual environment when done
    $ deactivate

Converting to End-to-End ONNX
-----------------------------

Since YOLOv10 is based on Ultralytics' framework, we can use the same conversion
process as YOLOv8:

.. code-block:: console

    # Clone the YOLOv8-TensorRT repository
    $ git clone https://github.com/triple-Mu/YOLOv8-TensorRT.git
    $ cd YOLOv8-TensorRT

    # Convert the weights to end-to-end format
    # Adjust parameters according to your needs:
    # - iou-thres: IoU threshold for NMS
    # - conf-thres: Confidence threshold
    # - topk: Maximum number of detections
    # - input-shape: Input image size
    $ python3 export-det.py \
        --weights PATH_TO_WEIGHTS \
        --iou-thres 0.5 \
        --conf-thres 0.25 \
        --topk 100 \
        --opset 13 \
        --sim \
        --input-shape 1,3,640,640 \
        --device cuda:0

Building TensorRT Engine
------------------------

Once you have the end-to-end ONNX weights, build a TensorRT engine:

.. code-block:: python

    from trtutils.trtexec import build_engine

    # Build the engine with FP16 precision
    build_engine(
        weights="yolov10.onnx",
        output="yolov10.engine",
        fp16=True,
        workspace_size=1 << 30,  # 1GB workspace
    )

    # For Jetson devices with DLA support
    build_engine(
        weights="yolov10.onnx",
        output="yolov10_dla.engine",
        fp16=True,
        dla_core=0,  # Use DLA core 0
        workspace_size=1 << 30,
    )

Running Inference
-----------------

The :py:class:`~trtutils.impls.yolo.YOLO` class provides a high-level interface
for running YOLOv10 inference:

.. code-block:: python

    import cv2
    from trtutils.impls.yolo import YOLO, YOLO10

    # Load the YOLOv10 model
    yolo = YOLO("yolov10.engine")

    # OR, use the YOLO10 class
    yolo = YOLO10("yolov10.engine")

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

You can run multiple YOLOv10 models in parallel:

.. code-block:: python

    from trtutils.impls.yolo import ParallelYOLO

    # Create a parallel YOLO instance with multiple engines
    yolo = ParallelYOLO(["yolov10_1.engine", "yolov10_2.engine"])

    # Run inference on multiple images
    images = [cv2.imread(f"image{i}.jpg") for i in range(2)]
    results = yolo.end2end(images)

Benchmarking
^^^^^^^^^^^^

Measure performance with the built-in benchmarking utilities:

.. code-block:: python

    from trtutils import benchmark_engine

    # Run 1000 iterations
    results = benchmark_engine("yolov10.engine", iterations=1000)
    print(f"Average latency: {results.latency.mean:.2f}ms")
    print(f"Throughput: {1000/results.latency.mean:.2f} FPS")

    # On Jetson devices, measure power consumption
    from trtutils.jetson import benchmark_engine as jetson_benchmark

    results = jetson_benchmark(
        "yolov10.engine",
        iterations=1000,
        tegra_interval=1  # More frequent power measurements
    )
    print(f"Average power draw: {results.power_draw.mean:.2f}W")
    print(f"Total energy used: {results.energy.sum:.2f}J")

Troubleshooting
---------------

Common issues and solutions:

1. **ONNX Export Fails**
   - Ensure you're using the correct virtual environment
   - Check if your PyTorch weights are valid
   - Try different ONNX opset versions

2. **Engine Creation Fails**
   - Ensure you have enough GPU memory (workspace_size parameter)
   - Check if the ONNX weights are valid

3. **Incorrect Detections**
   - Verify the input image preprocessing matches the training
   - Check if the confidence and IoU thresholds are appropriate

4. **Performance Issues**
   - Try enabling FP16 precision
   - On Jetson devices, ensure MAXN power mode and enable jetson_clocks
