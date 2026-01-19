.. _tutorials_deimv2:

DEIMv2 Tutorial
===============

This tutorial will guide you through using trtutils with DEIMv2 models.
We will cover:

1. Downloading ONNX weights from DEIMv2
2. Building a TensorRT engine
3. Running inference with the engine
4. Advanced features and optimizations

Downloading ONNX Weights
-------------------------

DEIMv2 models can be automatically downloaded and converted to ONNX format using the trtutils CLI:

.. code-block:: bash

    # Download and convert DEIMv2 models to ONNX
    # Available models: deimv2_atto, deimv2_femto, deimv2_pico, deimv2_n, deimv2_s, deimv2_m, deimv2_l, deimv2_x
    $ python3 -m trtutils download --model deimv2_atto --output deimv2_atto.onnx --imgsz 640 --opset 17

    # For other DEIMv2 variants
    $ python3 -m trtutils download --model deimv2_n --output deimv2_n.onnx --imgsz 640 --opset 17

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
        onnx="deimv2_atto.onnx",
        output="deimv2_atto.engine",
        fp16=True,
        shapes=[("images", (1, 3, 640, 640)), ("orig_image_size", (1, 2))],
    )

    # For Jetson devices with DLA support
    build_engine(
        onnx="deimv2_atto.onnx",
        output="deimv2_atto_dla.engine",
        int8=True,  # Orin series optimize for int8
        fp16=True,  # Can use fp16 on Xavier series
        dla_core=0,  # Use DLA core 0
        shapes=[("images", (1, 3, 640, 640)), ("orig_image_size", (1, 2))],
    )

Running Inference
-----------------

The :py:class:`~trtutils.models.DEIMv2` class provides a high-level interface
for running DEIMv2 inference:

.. code-block:: python

    import cv2
    from trtutils.models import DEIMv2

    # Load the DEIMv2 model
    deimv2 = DEIMv2("deimv2_atto.engine")

    # Read and process an image
    img = cv2.imread("example.jpg")
    detections = deimv2.end2end(img)

    # Print results
    for bbox, confidence, class_id in detections:
        print(f"Class: {class_id}, Confidence: {confidence}")
        print(f"Bounding Box: {bbox}")
