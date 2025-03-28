.. _tutorials_yolo_v7:

YOLOv7 Tutorial
===============

This tutorial will guide you through using trtutils with YOLOv7 models.
We will cover:

1. Exporting ONNX weights from YOLOv7
2. Building a TensorRT engine
3. Running inference with the engine
4. Advanced features and optimizations

Exporting ONNX Weights
----------------------

YOLOv7 supports end-to-end export of ONNX weights directly. Here's how to do it:

.. code-block:: console

    # Clone the YOLOv7 repository
    $ git clone https://github.com/WongKinYiu/yolov7.git
    $ cd yolov7

    # Export the ONNX weights
    # Adjust parameters according to your needs:
    # - topk-all: Maximum number of detections
    # - iou-thres: IoU threshold for NMS
    # - conf-thres: Confidence threshold
    # - img-size: Input image size
    $ python3 export.py -weights PATH_TO_WEIGHTS \
                       --end2end \
                       --grid \
                       --simplify \
                       --topk-all 100 \
                       --iou-thres 0.5 \
                       --conf-thres 0.25 \
                       --img-size 640

Building TensorRT Engine
------------------------

Once you have the ONNX weights, you can build a TensorRT engine using trtutils:

.. code-block:: python

    from trtutils.trtexec import build_engine

    # Build the engine with FP16 precision
    build_engine(
        weights="yolov7.onnx",
        output="yolov7.engine",
        fp16=True,
        workspace_size=1 << 30,  # 1GB workspace
    )

    # For Jetson devices with DLA support
    build_engine(
        weights="yolov7.onnx",
        output="yolov7_dla.engine",
        int8=True,  # Orin series optimize for int8
        fp16=False,  # Can use fp16 on Xavier series
        dla_core=0,  # Use DLA core 0
        workspace_size=1 << 30,
    )

Running Inference
-----------------

The :py:class:`~trtutils.impls.yolo.YOLO` class provides a high-level interface
for running YOLOv7 inference:

.. code-block:: python

    import cv2
    from trtutils.impls.yolo import YOLO, YOLO7

    # Load the YOLOv7 model
    yolo = YOLO("yolov7.engine")

    # OR, use the YOLO7 class
    yolo = YOLO7("yolov7.engine")

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

You can run multiple YOLOv7 models in parallel:

.. code-block:: python

    from trtutils.impls.yolo import ParallelYOLO

    # Create a parallel YOLO instance with multiple engines
    yolo = ParallelYOLO(["yolov7_1.engine", "yolov7_2.engine"])

    # Run inference on multiple images
    images = [cv2.imread(f"image{i}.jpg") for i in range(2)]
    results = yolo.end2end(images)

Benchmarking
^^^^^^^^^^^^

Measure performance with the built-in benchmarking utilities:

.. code-block:: python

    from trtutils import benchmark_engine

    # Run 1000 iterations
    results = benchmark_engine("yolov7.engine", iterations=1000)
    print(f"Average latency: {results.latency.mean:.2f}ms")
    print(f"Throughput: {1000/results.latency.mean:.2f} FPS")

    # On Jetson devices, measure power consumption
    from trtutils.jetson import benchmark_engine as jetson_benchmark

    results = jetson_benchmark(
        "yolov7.engine",
        iterations=1000,
        tegra_interval=1  # More frequent power measurements
    )
    print(f"Average power draw: {results.power_draw.mean:.2f}W")
    print(f"Total energy used: {results.energy.sum:.2f}J")

Troubleshooting
---------------

Common issues and solutions:

1. **Engine Creation Fails**
   - Ensure you have enough GPU memory (workspace_size parameter)
   - Check if the ONNX weights are valid

2. **Incorrect Detections**
   - Verify the input image preprocessing matches the training
   - Check if the confidence and IoU thresholds are appropriate

3. **Performance Issues**
   - Try enabling FP16 precision
   - On Jetson devices, ensure MAXN power mode and enable jetson_clocks
