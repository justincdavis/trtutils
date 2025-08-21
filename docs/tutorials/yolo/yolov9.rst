.. _tutorials_yolo_v9:

YOLOv9 Tutorial
===============

This tutorial will guide you through using trtutils with YOLOv9 models.
We will cover:

1. Exporting ONNX weights from YOLOv9
2. Building a TensorRT engine
3. Running inference with the engine

Exporting ONNX Weights
----------------------

YOLOv9 is written by the same authors as YOLOv7 and supports similar exporting options.
However, it has a unique feature where the input size is explicitly marked as dynamic
in the ONNX weights. Here's how to export:

.. code-block:: bash

    # Clone the YOLOv9 repository
    $ git clone https://github.com/WongKinYiu/yolov9.git
    $ cd yolov9

    # Export the ONNX weights
    # Adjust parameters according to your needs:
    # - topk-all: Maximum number of detections
    # - iou-thres: IoU threshold for NMS
    # - conf-thres: Confidence threshold
    # - img-size: Input image size (will be dynamic in ONNX)
    $ python3 export.py \
        --weights PATH_TO_WEIGHTS \
        --include onnx_end2end \
        --simplify \
        --iou-thres 0.5 \
        --conf-thres 0.25 \
        --topk-all 100 \
        --img-size 640 640

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
