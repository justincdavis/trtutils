.. _models:

Supported Models
================

This page lists all models currently supported by trtutils.

Object Detection
-----------------

YOLO Models
~~~~~~~~~~~

1. **Ultralytics YOLO** - YOLOv8 and YOLOv11
   
   - YOLOv8: ``yolov8n``, ``yolov8s``, ``yolov8m``, ``yolov8l``, ``yolov8x``
   - YOLOv11: ``yolov11n``, ``yolov11s``, ``yolov11m``, ``yolov11l``, ``yolov11x``
   - GitHub: `ultralytics/ultralytics <https://github.com/ultralytics/ultralytics>`_

2. **YOLOv7**
   
   - GitHub: `WongKinYiu/yolov7 <https://github.com/WongKinYiu/yolov7>`_

3. **YOLOv9**
   
   - GitHub: `WongKinYiu/yolov9 <https://github.com/WongKinYiu/yolov9>`_

4. **YOLOv10**
   
   - GitHub: `THU-MIG/yolov10 <https://github.com/THU-MIG/yolov10>`_

5. **YOLOv12**
   
   - GitHub: `sunsmarterjie/yolov12 <https://github.com/sunsmarterjie/yolov12>`_

6. **YOLOv13**
   
   - GitHub: `iMoonLab/yolov13 <https://github.com/iMoonLab/yolov13>`_

7. **YOLOX**
   
   - GitHub: `Megvii-BaseDetection/YOLOX <https://github.com/Megvii-BaseDetection/YOLOX>`_

DETR Models
~~~~~~~~~~~

8. **RT-DETRv1**
   
   - GitHub: `lyuwenyu/RT-DETR <https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch>`_

9. **RT-DETRv2**
   
   - GitHub: `lyuwenyu/RT-DETR <https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch>`_

10. **RT-DETRv3**
    
    - GitHub: `clxia12/RT-DETRv3 <https://github.com/clxia12/RT-DETRv3>`_

11. **D-FINE**
    
    - GitHub: `Peterande/D-FINE <https://github.com/Peterande/D-FINE>`_

12. **DEIM**
    
    - GitHub: `Intellindust-AI-Lab/DEIM <https://github.com/Intellindust-AI-Lab/DEIM>`_

13. **DEIMv2**
    
    - GitHub: `Intellindust-AI-Lab/DEIMv2 <https://github.com/Intellindust-AI-Lab/DEIMv2>`_

14. **RF-DETR**
    
    - GitHub: `roboflow/rf-detr <https://github.com/roboflow/rf-detr>`_

Classification
--------------

1. **Torchvision Classifiers**

   - See: `PyTorch Vision Classification Models <https://docs.pytorch.org/vision/main/models.html#classification>`_

2. **Ultralytics YOLO** - YOLOv8, YOLOv11, and YOLOv26 classification heads

   - YOLOv8: ``yolov8n-cls``, ``yolov8s-cls``, ``yolov8m-cls``, ``yolov8l-cls``, ``yolov8x-cls``
   - YOLOv11: ``yolov11n-cls``, ``yolov11s-cls``, ``yolov11m-cls``, ``yolov11l-cls``, ``yolov11x-cls``
   - YOLOv26: ``yolov26n-cls``, ``yolov26s-cls``, ``yolov26m-cls``, ``yolov26l-cls``, ``yolov26x-cls``
   - GitHub: `ultralytics/ultralytics <https://github.com/ultralytics/ultralytics>`_
   - AGPL-3.0 and commercial licensed, be aware of license restrictions

Instance Segmentation
----------------------

1. **Ultralytics YOLO** - YOLOv8, YOLOv11, and YOLOv26 segmentation heads

   - YOLOv8: ``yolov8n-seg``, ``yolov8s-seg``, ``yolov8m-seg``, ``yolov8l-seg``, ``yolov8x-seg``
   - YOLOv11: ``yolov11n-seg``, ``yolov11s-seg``, ``yolov11m-seg``, ``yolov11l-seg``, ``yolov11x-seg``
   - YOLOv26: ``yolov26n-seg``, ``yolov26s-seg``, ``yolov26m-seg``, ``yolov26l-seg``, ``yolov26x-seg``
   - GitHub: `ultralytics/ultralytics <https://github.com/ultralytics/ultralytics>`_
   - AGPL-3.0 and commercial licensed, be aware of license restrictions

Pose Estimation
----------------

1. **Ultralytics YOLO** - YOLOv8, YOLOv11, and YOLOv26 pose heads

   - YOLOv8: ``yolov8n-pose``, ``yolov8s-pose``, ``yolov8m-pose``, ``yolov8l-pose``, ``yolov8x-pose``
   - YOLOv11: ``yolov11n-pose``, ``yolov11s-pose``, ``yolov11m-pose``, ``yolov11l-pose``, ``yolov11x-pose``
   - YOLOv26: ``yolov26n-pose``, ``yolov26s-pose``, ``yolov26m-pose``, ``yolov26l-pose``, ``yolov26x-pose``
   - GitHub: `ultralytics/ultralytics <https://github.com/ultralytics/ultralytics>`_
   - AGPL-3.0 and commercial licensed, be aware of license restrictions

Oriented Bounding Boxes
-------------------------

1. **Ultralytics YOLO** - YOLOv8, YOLOv11, and YOLOv26 OBB heads

   - YOLOv8: ``yolov8n-obb``, ``yolov8s-obb``, ``yolov8m-obb``, ``yolov8l-obb``, ``yolov8x-obb``
   - YOLOv11: ``yolov11n-obb``, ``yolov11s-obb``, ``yolov11m-obb``, ``yolov11l-obb``, ``yolov11x-obb``
   - YOLOv26: ``yolov26n-obb``, ``yolov26s-obb``, ``yolov26m-obb``, ``yolov26l-obb``, ``yolov26x-obb``
   - GitHub: `ultralytics/ultralytics <https://github.com/ultralytics/ultralytics>`_
   - AGPL-3.0 and commercial licensed, be aware of license restrictions

Note: YOLOv12 does not publish segmentation, classification, pose, or OBB weights upstream
(only detection), so it is absent from the three sections above and the classification list.

Hand-Object Interaction
------------------------

1. **HOI-DETR**

   - GitHub: `AhmadDarKhalil/HOI-DETR <https://github.com/AhmadDarKhalil/HOI-DETR>`_

2. **Hands23**

   - GitHub: `EvaCheng-cty/hands23_detector <https://github.com/EvaCheng-cty/hands23_detector>`_

Model Download Support
----------------------

The following models can be automatically downloaded and converted to ONNX format using the :ref:`download CLI command <cli>`:

YOLO Models
~~~~~~~~~~~

- **YOLOv7**: All variants with pretrained weights
- **YOLOv8**: ``yolov8n``, ``yolov8s``, ``yolov8m``, ``yolov8l``, ``yolov8x`` (via Ultralytics)
- **YOLOv9**: All variants with pretrained weights
- **YOLOv10**: All variants with pretrained weights
- **YOLOv11**: ``yolov11n``, ``yolov11s``, ``yolov11m``, ``yolov11l``, ``yolov11x`` (via Ultralytics)
- **YOLOv12**: All variants with pretrained weights
- **YOLOv13**: ``yolov13n``, ``yolov13s``, ``yolov13l``, ``yolov13x``
- **YOLOX**: ``yoloxn``, ``yoloxt``, ``yoloxs``, ``yoloxm``, ``yoloxl``, ``yoloxx``, ``yolox_darknet``

DETR Models
~~~~~~~~~~~

- **RT-DETRv1**: Multiple configurations available
- **RT-DETRv2**: Multiple configurations available
- **RT-DETRv3**: Multiple configurations available
- **D-FINE**: Multiple configurations available
- **DEIM**: Multiple configurations available
- **DEIMv2**: ``deimv2_atto``, ``deimv2_femto``, ``deimv2_pico``, ``deimv2_n``, ``deimv2_s``, ``deimv2_m``, ``deimv2_l``, ``deimv2_x``
- **RF-DETR**: Multiple configurations available

Hand-Object Interaction Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **HOI-DETR**: ``hoi_detr_vitl``
- **Hands23**: ``hands23_x101``

Instance Segmentation Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **YOLOv8-seg**: ``yolov8n-seg``, ``yolov8s-seg``, ``yolov8m-seg``, ``yolov8l-seg``, ``yolov8x-seg`` (via Ultralytics)
- **YOLOv11-seg**: ``yolov11n-seg``, ``yolov11s-seg``, ``yolov11m-seg``, ``yolov11l-seg``, ``yolov11x-seg`` (via Ultralytics)
- **YOLOv26-seg**: ``yolov26n-seg``, ``yolov26s-seg``, ``yolov26m-seg``, ``yolov26l-seg``, ``yolov26x-seg`` (via Ultralytics)

Pose Estimation Models
~~~~~~~~~~~~~~~~~~~~~~~

- **YOLOv8-pose**: ``yolov8n-pose``, ``yolov8s-pose``, ``yolov8m-pose``, ``yolov8l-pose``, ``yolov8x-pose`` (via Ultralytics)
- **YOLOv11-pose**: ``yolov11n-pose``, ``yolov11s-pose``, ``yolov11m-pose``, ``yolov11l-pose``, ``yolov11x-pose`` (via Ultralytics)
- **YOLOv26-pose**: ``yolov26n-pose``, ``yolov26s-pose``, ``yolov26m-pose``, ``yolov26l-pose``, ``yolov26x-pose`` (via Ultralytics)

Oriented Bounding Box Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **YOLOv8-obb**: ``yolov8n-obb``, ``yolov8s-obb``, ``yolov8m-obb``, ``yolov8l-obb``, ``yolov8x-obb`` (via Ultralytics)
- **YOLOv11-obb**: ``yolov11n-obb``, ``yolov11s-obb``, ``yolov11m-obb``, ``yolov11l-obb``, ``yolov11x-obb`` (via Ultralytics)
- **YOLOv26-obb**: ``yolov26n-obb``, ``yolov26s-obb``, ``yolov26m-obb``, ``yolov26l-obb``, ``yolov26x-obb`` (via Ultralytics)

Example Usage
~~~~~~~~~~~~~

Download models using the CLI:

.. code-block:: console

    $ python -m trtutils download --model yolov8n --output yolov8n.onnx
    $ python -m trtutils download --model yolov11m --output yolov11m.onnx --imgsz 640 --opset 17
    $ python -m trtutils download --model yoloxs --output yoloxs.onnx --imgsz 640 --opset 17

For more information on the download command, see the :ref:`CLI Reference <cli>`.
