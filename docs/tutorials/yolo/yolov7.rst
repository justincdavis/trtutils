.. _tutorials_yolo_v7:

YOLOv7 Tutorial
===============

This tutorial will guide you through using trtutils with YOLOv7 models.
We will cover:

1. Exporting ONNX weights from YOLOv7
2. Building a TensorRT engine
3. Running inference with the engine
4. Advanced features and optimizations

Exporting ONNX Weights
----------------------

YOLOv7 supports end-to-end export of ONNX weights directly. Here's how to do it:

.. code-block:: bash

    # Clone the YOLOv7 repository
    $ git clone https://github.com/WongKinYiu/yolov7.git
    $ cd yolov7

    # Export the ONNX weights
    # Adjust parameters according to your needs:
    # - topk-all: Maximum number of detections
    # - iou-thres: IoU threshold for NMS
    # - conf-thres: Confidence threshold
    # - img-size: Input image size
    $ python3 export.py -weights PATH_TO_WEIGHTS \
                       --end2end \
                       --grid \
                       --simplify \
                       --topk-all 100 \
                       --iou-thres 0.5 \
                       --conf-thres 0.25 \
                       --img-size 640

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
        onnx="yolov7.onnx",
        output="yolov7.engine",
        fp16=True,
    )

    # For Jetson devices with DLA support
    build_engine(
        onnx="yolov7.onnx",
        output="yolov7_dla.engine",
        int8=True,  # Orin series optimize for int8
        fp16=True,  # Can use fp16 on Xavier series
        dla_core=0,  # Use DLA core 0
    )

Running Inference
-----------------

The :py:class:`~trtutils.models.YOLO` class provides a high-level interface
for running YOLOv7 inference:

.. code-block:: python

    import cv2
    from trtutils.models import YOLO, YOLO7

    # Load the YOLOv7 model
    yolo = YOLO("yolov7.engine")

    # OR, use the YOLO7 class
    yolo = YOLO7("yolov7.engine")

    # Read and process an image
    img = cv2.imread("example.jpg")
    detections = yolo.end2end(img)

    # Print results
    for bbox, confidence, class_id in detections:
        print(f"Class: {class_id}, Confidence: {confidence}")
        print(f"Bounding Box: {bbox}")
