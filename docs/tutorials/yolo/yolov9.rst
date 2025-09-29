.. _tutorials_yolo_v9:

YOLOv9 Tutorial
===============

This tutorial will guide you through using trtutils with YOLOv9 models.
We will cover:

1. Downloading ONNX weights from YOLOv9
2. Building a TensorRT engine
3. Running inference with the engine

Downloading ONNX Weights
-------------------------

YOLOv9 models can be automatically downloaded and converted to ONNX format using the trtutils CLI:

.. code-block:: bash

    # Download and convert YOLOv9 models to ONNX
    # Available models: yolov9, yolov9-c, yolov9-e, yolov9-m, yolov9-s, yolov9-t
    $ python3 -m trtutils download --model yolov9 --output yolov9.onnx --imgsz 640 --opset 17

    # For other YOLOv9 variants
    $ python3 -m trtutils download --model yolov9-c --output yolov9-c.onnx --imgsz 640 --opset 17

Building TensorRT Engine
------------------------

When building the TensorRT engine for YOLOv9, you need to explicitly specify
the input shape since it's dynamic in the ONNX weights:

.. code-block:: python

    from trtutils.builder import build_engine

    # Build the engine with FP16 precision
    # Note: The input shape must match the img-size used during export
    build_engine(
        onnx="yolov9.onnx",
        output="yolov9.engine",
        fp16=True,
        shapes=[("images", (1, 3, 640, 640))],  # Must match export img-size
    )

    # For Jetson devices with DLA support
    build_engine(
        onnx="yolov9.onnx",
        output="yolov9_dla.engine",
        fp16=True,
        shapes=[("images", (1, 3, 640, 640))],
        dla_core=0,  # Use DLA core 0
    )

Running Inference
-----------------

The :py:class:`~trtutils.models.YOLO` class provides a high-level interface
for running YOLOv9 inference:

.. code-block:: python

    import cv2
    from trtutils.models import YOLO, YOLO9

    # Load the YOLOv9 model
    yolo = YOLO("yolov9.engine")

    # OR, use the YOLO9 class
    yolo = YOLO9("yolov9.engine")

    # Read and process an image
    img = cv2.imread("example.jpg")
    detections = yolo.end2end(img)

    # Print results
    for bbox, confidence, class_id in detections:
        print(f"Class: {class_id}, Confidence: {confidence}")
        print(f"Bounding Box: {bbox}")
