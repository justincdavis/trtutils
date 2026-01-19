.. _tutorials_rtdetrv3:

RT-DETRv3 Tutorial
==================

This tutorial will guide you through using trtutils with RT-DETRv3 models.
We will cover:

1. Downloading ONNX weights from RT-DETRv3
2. Building a TensorRT engine
3. Running inference with the engine
4. Advanced features and optimizations

Downloading ONNX Weights
-------------------------

RT-DETRv3 models can be automatically downloaded and converted to ONNX format using the trtutils CLI:

.. code-block:: bash

    # Download and convert RT-DETRv3 models to ONNX
    # Available models: rtdetrv3_r18, rtdetrv3_r34, rtdetrv3_r50, rtdetrv3_r101
    $ python3 -m trtutils download --model rtdetrv3_r18 --output rtdetrv3_r18.onnx --imgsz 640 --opset 16

    # For other RT-DETRv3 variants
    $ python3 -m trtutils download --model rtdetrv3_r50 --output rtdetrv3_r50.onnx --imgsz 640 --opset 16

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
        --shape image:1,3,640,640 \
        --shape im_shape:1,2 \
        --shape scale_factor:1,2

Alternatively, if you want to export the engine using the Python API:

.. code-block:: python

    from trtutils.builder import build_engine

    # Build the engine with FP16 precision
    build_engine(
        onnx="rtdetrv3_r18.onnx",
        output="rtdetrv3_r18.engine",
        fp16=True,
        shapes=[("image", (1, 3, 640, 640)), ("im_shape", (1, 2)), ("scale_factor", (1, 2))],
    )

    # For Jetson devices with DLA support
    build_engine(
        onnx="rtdetrv3_r18.onnx",
        output="rtdetrv3_r18_dla.engine",
        int8=True,  # Orin series optimize for int8
        fp16=True,  # Can use fp16 on Xavier series
        dla_core=0,  # Use DLA core 0
        shapes=[("image", (1, 3, 640, 640)), ("im_shape", (1, 2)), ("scale_factor", (1, 2))],
    )
