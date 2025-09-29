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

YOLOv11 is implemented by Ultralytics, and can be exported from Ultralytics directly:

.. code-block:: bash

    # Install ultralytics if you haven't already
    $ pip install ultralytics
    
    # Export ONNX weights
    # This will save the ONNX file in the same directory as your PyTorch weights
    $ yolo export model=TORCH_WEIGHTS format=onnx simplify=True

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
        onnx="yolov11.onnx",
        output="yolov11.engine",
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
for running YOLOv11 inference:

.. code-block:: python

    import cv2
    from trtutils.models import YOLO, YOLO11

    # Load the YOLOv11 model
    yolo = YOLO("yolov11.engine")

    # OR, use the YOLO11 class
    yolo = YOLO11("yolov11.engine")

    # Read and process an image
    img = cv2.imread("example.jpg")
    detections = yolo.end2end(img)

    # Print results
    for bbox, confidence, class_id in detections:
        print(f"Class: {class_id}, Confidence: {confidence}")
        print(f"Bounding Box: {bbox}")
