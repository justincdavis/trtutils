.. _tutorial_overview:

Tutorial Overview
=================

This section contains comprehensive tutorials for using trtutils. Each tutorial
provides step-by-step instructions and examples for different use cases.

.. toctree::
    :maxdepth: 1
    :caption: Contents:

    Basic Usage <basic>
    Advanced Usage <advanced>
    YOLO Basic Usage <yolo>
    YOLO Tutorials <yolo/index>
    RT-DETR Tutorials <rtdetr/index>

Basic Tutorials
---------------

The basic tutorials cover essential functionality:

- Basic Usage: Learn how to use the core TRTEngine and TRTModel classes
- Advanced Usage: Explore lower-level interfaces and CUDA operations
- YOLO Basic Usage: Learn how to use the YOLO class

YOLO Tutorials
--------------

The YOLO tutorials provide detailed instructions for working with different YOLO variants:

- YOLOv7: Direct end-to-end ONNX export
- YOLOv8: Two-step ONNX conversion process
- YOLOv9: Dynamic input shape handling
- YOLOv10: Ultralytics-based implementation
- YOLOv11: Ultralytics-based implementation
- YOLOv12: Ultralytics-based implementation
- YOLOv13: Ultralytics-based implementation
- YOLOX: Special input range handling

Each YOLO tutorial covers:
- ONNX weight download and conversion
- TensorRT engine building
- Inference with the YOLO class
- Advanced features and optimizations
- Troubleshooting guides

RT-DETR Tutorials
-----------------

The RT-DETR tutorials provide detailed instructions for working with different RT-DETR variants and related models:

- RT-DETRv1: Direct end-to-end ONNX export
- RT-DETRv2: Direct end-to-end ONNX export
- RT-DETRv3: PaddlePaddle-based conversion
- D-FINE: High performance detection
- DEIM: Efficient detection model
- DEIMv2: Improved version of DEIM
- RF-DETR: Roboflow's DETR implementation

Each RT-DETR tutorial covers:
- ONNX weight download and conversion
- TensorRT engine building
- Inference with the appropriate model class
- Advanced features and optimizations
- Troubleshooting guides

For more information about specific YOLO variants, see the :ref:`YOLO Tutorials <tutorials_yolo>` section.
For more information about specific RT-DETR variants, see the :ref:`RT-DETR Tutorials <tutorials_rtdetr>` section.
