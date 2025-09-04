.. _tutorials_rtdetrv3:

RT-DETRv3 Tutorial
==================

This tutorial will guide you through using trtutils with RT-DETRv3 models.
We will cover:

1. Exporting ONNX weights from RT-DETRv3
2. Building a TensorRT engine
3. Running inference with the engine
4. Advanced features and optimizations

Exporting ONNX Weights
----------------------

RT-DETRv3 supports end-to-end export of ONNX weights directly. Here's how to do it:

.. code-block:: bash

    # Clone the RT-DETRv3 repository
    $ git clone https://github.com/clxia12/RT-DETRv3
    $ cd RT-DETRv3

    # Install requirements
    $ python3 -m venv .venv
    $ source .venv/bin/activate
    $ pip install -r requirements.txt paddlepaddle==2.6.1 paddle2onnx==1.0.5 onnx==1.13.0

    # Assumes you have already downloaded some weights

    # Export the Paddle weights
    # Adjust the config based on the weights you downloaded
    $ python3 tools/export_model.py -c $CONFIG_PATH -o weights=$WEIGHTS_PATH use_gpu=false trt=True --output_dir=output_inference
    # Example using actual weight names
    # $ python3 tools/export_model.py -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
    #          -o weights="$WEIGHTS_PATH" use_gpu=false trt=True \
    #          --output_dir=output_inference
    # This saves to output_inference/rtdetrv3_r18vd_6x_coco/model.onnx

    # Export ONNX weights
    $ .venv/bin/paddle2onnx --model_dir=output_inference/WEIGHTS_PATH \
        --model_filename model.pdmodel \
        --params_filename model.pdiparams \
        --opset_version 16 \
        --save_file rtdetrv3.onnx

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
