.. _tutorials_rfdetr:

RF-DETR Tutorial
================

This tutorial will guide you through using trtutils with RF-DETR models.
We will cover:

1. Downloading ONNX weights from RF-DETR
2. Building a TensorRT engine
3. Running inference with the engine
4. Advanced features and optimizations

Downloading ONNX Weights
-------------------------

RF-DETR models can be automatically downloaded and converted to ONNX format using the trtutils CLI:

.. code-block:: bash

    # Download and convert RF-DETR models to ONNX
    # Available models: rfdetr_n, rfdetr_s, rfdetr_m
    $ python3 -m trtutils download --model rfdetr_n --output rfdetr_n.onnx --imgsz 640 --opset 17

    # For other RF-DETR variants
    $ python3 -m trtutils download --model rfdetr_s --output rfdetr_s.onnx --imgsz 640 --opset 17

Building TensorRT Engine
------------------------

Once you have the ONNX weights, you can build a TensorRT engine using trtutils:

.. code-block:: bash

    # Note we can build directly from the ONNX weights
    # Note: Need to specify the input shapes (namely the batch dimension is
    # left dynamic)
    python3 -m trtutils build \
        --onnx $ONNX_PATH \
        --output $OUTPUT_PATH \
        --fp16 \
        --shape images:1,3,640,640 \
        --shape orig_image_size:1,2

Alternatively, if you want to export the engine using the Python API:

.. code-block:: python

    from trtutils.builder import build_engine

    # Build the engine with FP16 precision
    build_engine(
        onnx="rfdetr_n.onnx",
        output="rfdetr_n.engine",
        fp16=True,
        shapes=[("images", (1, 3, 640, 640)), ("orig_image_size", (1, 2))],
    )

    # For Jetson devices with DLA support
    build_engine(
        onnx="rfdetr_n.onnx",
        output="rfdetr_n_dla.engine",
        int8=True,  # Orin series optimize for int8
        fp16=True,  # Can use fp16 on Xavier series
        dla_core=0,  # Use DLA core 0
        shapes=[("images", (1, 3, 640, 640)), ("orig_image_size", (1, 2))],
    )

Running Inference
-----------------

The :py:class:`~trtutils.models.RFDETR` class provides a high-level interface
for running RF-DETR inference:

.. code-block:: python

    import cv2
    from trtutils.models import RFDETR

    # Load the RF-DETR model
    rfdetr = RFDETR("rfdetr_n.engine")

    # Read and process an image
    img = cv2.imread("example.jpg")
    detections = rfdetr.end2end(img)

    # Print results
    for bbox, confidence, class_id in detections:
        print(f"Class: {class_id}, Confidence: {confidence}")
        print(f"Bounding Box: {bbox}")