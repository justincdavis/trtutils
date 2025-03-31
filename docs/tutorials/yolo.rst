.. _basic_yolo:

YOLO Basic Usage
================

This tutorial will guide you through using trtutils with YOLO models.
We will cover:

1. Building a TensorRT engine from ONNX weights
2. Running inference with the engine
3. Advanced features like parallel execution and benchmarking

Building TensorRT Engines
-------------------------

trtutils provides a simple interface to build TensorRT engines from ONNX weights.
The :py:func:`~trtutils.trtexec.find_trtexec` function will locate your trtexec
installation and :py:func:`~trtutils.trtexec.build_engine` will build the engine.

Example:

.. code-block:: python

    from trtutils.trtexec import build_engine

    # Build a TensorRT engine from ONNX weights
    build_engine(
        "yolo.onnx",
        "yolo.engine",
        precision="fp16",
        workspace_size=1 << 30,  # 1GB
    )

Running YOLO Inference
----------------------

The :py:class:`~trtutils.impls.yolo.YOLO` class provides a high-level interface
for running YOLO inference. It handles all the preprocessing and postprocessing
steps automatically.

Example:

.. code-block:: python

    import cv2
    from trtutils.impls.yolo import YOLO

    # Load the YOLO model
    yolo = YOLO("yolo.engine")

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

You can run multiple YOLO models in parallel using the :py:class:`~trtutils.impls.yolo.ParallelYOLO` class:

.. code-block:: python

    from trtutils.impls.yolo import ParallelYOLO

    # Create a parallel YOLO instance with multiple engines
    yolo = ParallelYOLO(["yolo1.engine", "yolo2.engine"])

    # Run inference on multiple images
    images = [cv2.imread(f"image{i}.jpg") for i in range(2)]
    results = yolo.end2end(images)

    # OR
    yolo.submit(images)
    results = yolo.retrieve()

    # OR
    yolo.submit_model(images[0], 0)
    single_result = yolo.retrieve_model(0)

    # print results
    for i, result in enumerate(results):
        print(f"Results for model {i}:")
        for bbox, confidence, class_id in result:
            print(f"Class: {class_id}, Confidence: {confidence}")
            print(f"Bounding Box: {bbox}")

Benchmarking
^^^^^^^^^^^^

You can benchmark YOLO models using the built-in benchmarking utilities:

.. code-block:: python

    from trtutils import benchmark_engine

    # Run 1000 iterations
    results = benchmark_engine("yolo.engine", iterations=1000)
    print(f"Average latency: {results.latency.mean:.2f}ms")
    print(f"Throughput: {1000/results.latency.mean:.2f} FPS")

    # On Jetson devices, you can also measure power consumption
    from trtutils.jetson import benchmark_engine as jetson_benchmark

    results = jetson_benchmark(
        "yolo.engine",
        iterations=1000,
        tegra_interval=1  # More frequent power measurements
    )
    print(f"Average power draw: {results.power_draw.mean:.2f}W")
    print(f"Total energy used: {results.energy.mean:.2f}J")
