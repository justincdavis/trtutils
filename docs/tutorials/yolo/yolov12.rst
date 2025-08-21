.. _tutorials_yolo_v11:

YOLOv11 Tutorial
================

This tutorial will guide you through using trtutils with YOLOv11 models.
We will cover:

1. Exporting ONNX weights from YOLOv11
2. Building a TensorRT engine
3. Running inference with the engine

Exporting ONNX Weights
----------------------

YOLOv10 is built on top of Ultralytics' framework and requires a virtual environment
to avoid conflicts with other packages. Here's how to export the ONNX weights:

.. code-block:: bash

    # Clone the YOLOv12 repository
    $ git clone https://github.com/sunsmarterjie/yolov12
    $ cd yolov12

    # Create and activate a virtual environment
    $ python3 -m venv yolov12
    $ source yolov12/bin/activate

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

Once you have the ONNX weights, build a TensorRT engine:

.. code-block:: bash

    # build_yolo is an alias for the 'build' command with '--yolo' passed to it
    python3 -m trtutils build_yolo \
        --onnx PATH_TO_WEIGHTS \
        --output PATH_TO_OUTPUT \
        --fp16 \
        --num_classes 80 \
        --iou_threshold 0.5 \
        --conf_threshold 0.25 \
        --top_k 100

Alternatively, if you want to export the engine using the Python API:

.. code-block:: python

    from trtutils.builder import build_engine, hooks

    build_engine(
        onnx="yolov12.onnx",
        output="yolov12.engine",
        fp16=True,
        hooks=[hooks.yolo_efficient_nms_hook(
            num_classes=80,
            iou_threshold=0.5,
            conf_threshold=0.25,
            top_k=100,
        )]
    )

Running Inference
-----------------

The :py:class:`~trtutils.models.YOLO` class provides a high-level interface
for running YOLOv12 inference:

.. code-block:: python

    import cv2
    from trtutils.models import YOLO, YOLO12

    # Load the YOLO12 model
    yolo = YOLO("yolov12.engine")

    # OR, use the YOLO12 class
    yolo = YOLO12("yolov12.engine")

    # Read and process an image
    img = cv2.imread("example.jpg")
    detections = yolo.end2end(img)

    # Print results
    for bbox, confidence, class_id in detections:
        print(f"Class: {class_id}, Confidence: {confidence}")
        print(f"Bounding Box: {bbox}")
