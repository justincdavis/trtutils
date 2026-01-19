.. _tutorials_yolo_v13:

YOLOv13 Tutorial
================

This tutorial will guide you through using trtutils with YOLOv13 models.
We will cover:

1. Downloading ONNX weights from YOLOv13
2. Building a TensorRT engine
3. Running inference with the engine

Downloading ONNX Weights
-------------------------

YOLOv13 models can be automatically downloaded and converted to ONNX format using the trtutils CLI:

.. code-block:: bash

    # Download and convert YOLOv13 models to ONNX
    # Available models: yolov13n, yolov13s, yolov13l, yolov13x
    $ python3 -m trtutils download --model yolov13n --output yolov13n.onnx --imgsz 640 --opset 17

    # For other YOLOv13 variants
    $ python3 -m trtutils download --model yolov13s --output yolov13s.onnx --imgsz 640 --opset 17

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
        onnx="yolov13.onnx",
        output="yolov13.engine",
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
for running YOLOv13 inference:

.. code-block:: python

    import cv2
    from trtutils.models import YOLO, YOLO13

    # Load the YOLOv13 model
    yolo = YOLO("yolov13.engine")

    # OR, use the YOLO13 class
    yolo = YOLO13("yolov13.engine")

    # Read and process an image
    img = cv2.imread("example.jpg")
    detections = yolo.end2end(img)

    # Print results
    for bbox, confidence, class_id in detections:
        print(f"Class: {class_id}, Confidence: {confidence}")
        print(f"Bounding Box: {bbox}")
