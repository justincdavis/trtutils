.. _tutorials_yolo_v10:

YOLOv10 Tutorial
================

This tutorial will guide you through using trtutils with YOLOv10 models.
We will cover:

1. Exporting ONNX weights from YOLOv10
2. Building a TensorRT engine
3. Running inference with the engine

Exporting ONNX Weights
----------------------

YOLOv10 is built on top of Ultralytics' framework and requires a virtual environment
to avoid conflicts with other packages. Here's how to export the ONNX weights:

.. code-block:: bash

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
        model=PATH_TO_WEIGHTS \
        format=onnx \
        opset=13 \
        simplify \
        imgsz=640

    # Deactivate the virtual environment when done
    $ deactivate

Building TensorRT Engine
------------------------

Once you have the ONNX weights, you can build a TensorRT engine using trtutils:

.. code-block:: bash

    # Note that build_yolo is not used since we exported the end2end model
    # using the ONNX weights directly
    python3 -m trtutils build \
        --onnx PATH_TO_WEIGHTS \
        --output PATH_TO_OUTPUT \
        --fp16

Alternatively, if you want to export the engine using the Python API:

.. code-block:: python

    from trtutils.builder import build_engine

    # Build the engine with FP16 precision
    build_engine(
        onnx="yolov10.onnx",
        output="yolov10.engine",
        fp16=True,
    )

    # For Jetson devices with DLA support
    build_engine(
        onnx="yolov10.onnx",
        output="yolov10_dla.engine",
        int8=True,  # Orin series optimize for int8
        fp16=True,  # Can use fp16 on Xavier series
        dla_core=0,  # Use DLA core 0
    )

Running Inference
-----------------

The :py:class:`~trtutils.models.YOLO` class provides a high-level interface
for running YOLOv10 inference:

.. code-block:: python

    import cv2
    from trtutils.models import YOLO, YOLO10

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
