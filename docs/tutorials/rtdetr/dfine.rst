.. _tutorials_dfine:

D-FINE Tutorial
===============

This tutorial will guide you through using trtutils with D-FINE models.
We will cover:

1. Downloading ONNX weights from D-FINE
2. Building a TensorRT engine
3. Running inference with the engine
4. Advanced features and optimizations

Downloading ONNX Weights
-------------------------

D-FINE models can be automatically downloaded and converted to ONNX format using the trtutils CLI:

.. code-block:: bash

    # Download and convert D-FINE models to ONNX
    # Available models: dfine_n, dfine_s, dfine_m, dfine_l, dfine_x
    $ python3 -m trtutils download --model dfine_n --output dfine_n.onnx --imgsz 640 --opset 17

    # For other D-FINE variants
    $ python3 -m trtutils download --model dfine_s --output dfine_s.onnx --imgsz 640 --opset 17

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
        onnx="dfine_n.onnx",
        output="dfine_n.engine",
        fp16=True,
        shapes=[("images", (1, 3, 640, 640)), ("orig_image_size", (1, 2))],
    )

    # For Jetson devices with DLA support
    build_engine(
        onnx="dfine_n.onnx",
        output="dfine_n.engine",
        int8=True,  # Orin series optimize for int8
        fp16=True,  # Can use fp16 on Xavier series
        dla_core=0,  # Use DLA core 0
        shapes=[("images", (1, 3, 640, 640)), ("orig_image_size", (1, 2))],
    )
