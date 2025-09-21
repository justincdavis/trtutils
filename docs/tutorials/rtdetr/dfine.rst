.. _tutorials_dfine:

D-FINE Tutorial
===============

This tutorial will guide you through using trtutils with D-FINE models.
We will cover:

1. Exporting ONNX weights from D-FINE
2. Building a TensorRT engine
3. Running inference with the engine
4. Advanced features and optimizations

Exporting ONNX Weights
----------------------

D-FINE supports end-to-end export of ONNX weights directly. Here's how to do it:

.. code-block:: bash

    # Clone the D-FINE repository
    $ git clone https://github.com/Peterande/D-FINE.git
    $ cd D-FINE

    # Assumes you have already downloaded some weights

    # Export the ONNX weights
    # Adjust the config based on the weights you downloaded
    $ python3 tools/deployment/export_onnx.py -c $CONFIG_PATH -r $WEIGHTS_PATH --check --simplify
    # Example using actual weight names
    # $ python3 tools/export_onnx.py -c configs/dfine/dfine_hgnetv2_n_coco.yml -r dfine_n_coco.pthvd_120e_coco_from_paddle.pth --check --simplify
    # This saves to dfine_n_coco.onnx

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
