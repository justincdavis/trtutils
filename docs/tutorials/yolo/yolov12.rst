.. _tutorials_yolo_v12:

YOLOv12 Tutorial
================

This tutorial will guide you through using trtutils with YOLOv12 models.
We will cover:

1. Downloading ONNX weights from YOLOv12
2. Building a TensorRT engine
3. Running inference with the engine

Downloading ONNX Weights
-------------------------

YOLOv12 models can be automatically downloaded and converted to ONNX format using the trtutils CLI:

.. code-block:: bash

    # Download and convert YOLOv12 models to ONNX
    # Available models: yolov12n, yolov12s, yolov12m, yolov12l, yolov12x
    $ python3 -m trtutils download --model yolov12n --output yolov12n.onnx --imgsz 640 --opset 17

    # For other YOLOv12 variants
    $ python3 -m trtutils download --model yolov12s --output yolov12s.onnx --imgsz 640 --opset 17

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

    # Load the YOLOv12 model
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
