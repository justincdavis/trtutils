.. _tutorials_yolo_x:

YOLOX Tutorial
==============

This tutorial will guide you through using trtutils with YOLOX models.
We will cover:

1. Downloading ONNX weights from YOLOX
2. Building a TensorRT engine
3. Running inference with the engine

Downloading ONNX Weights
-------------------------

YOLOX models can be automatically downloaded and converted to ONNX format using the trtutils CLI:

.. code-block:: bash

    # Download and convert YOLOX models to ONNX
    # Available models: yoloxn, yoloxt, yoloxs, yoloxm, yoloxl, yoloxx, yolox_darknet
    $ python3 -m trtutils download --model yoloxs --output yoloxs.onnx --imgsz 640 --opset 17 --accept

    # For other YOLOX variants
    $ python3 -m trtutils download --model yoloxm --output yoloxm.onnx --imgsz 640 --opset 17 --accept

Building TensorRT Engine
------------------------

Once you have the ONNX weights, build a TensorRT engine:

.. code-block:: bash

    python3 -m trtutils build_yolo \
        --weights PATH_TO_WEIGHTS \
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
        weights="yolox.onnx",
        output="yolox.engine",
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
for running YOLOX inference. Note that YOLOX requires input images to be in
the range [0, 255]:

.. code-block:: python

    import cv2
    from trtutils.models import YOLO, YOLOX

    # Load the YOLOX model with correct input range
    yolo = YOLO("yolox.engine", input_range=(0, 255))

    # OR, use the YOLOX class
    yolo = YOLOX("yolox.engine")

    # Read and process an image
    img = cv2.imread("example.jpg")
    detections = yolo.end2end(img)

    # Print results
    for bbox, confidence, class_id in detections:
        print(f"Class: {class_id}, Confidence: {confidence}")
        print(f"Bounding Box: {bbox}")
