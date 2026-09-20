.. _models:

Supported Models
================

This page lists all models currently supported by trtutils.

Object Detection
-----------------

YOLO Models
~~~~~~~~~~~

1. **Ultralytics YOLO** - YOLOv3, YOLOv5, YOLOv8, YOLOv11, and YOLOv26

   - YOLOv3: ``yolov3tu``, ``yolov3u``, ``yolov3sppu``
   - YOLOv5: ``yolov5nu``, ``yolov5su``, ``yolov5mu``, ``yolov5lu``, ``yolov5xu`` (and the ``6u`` variants)
   - YOLOv8: ``yolov8n``, ``yolov8s``, ``yolov8m``, ``yolov8l``, ``yolov8x``
   - YOLOv11: ``yolov11n``, ``yolov11s``, ``yolov11m``, ``yolov11l``, ``yolov11x``
   - YOLOv26: ``yolov26n``, ``yolov26s``, ``yolov26m``, ``yolov26l``, ``yolov26x``
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

Depth Estimation
----------------

1. **Depth-Anything-V1**

   - GitHub: `LiheYoung/Depth-Anything <https://github.com/LiheYoung/Depth-Anything>`_

2. **Depth-Anything-V2**

   - GitHub: `DepthAnything/Depth-Anything-V2 <https://github.com/DepthAnything/Depth-Anything-V2>`_

3. **Depth-Anything-V3**

   - GitHub: `ByteDance-Seed/depth-anything-3 <https://github.com/ByteDance-Seed/depth-anything-3>`_

Classification
--------------

1. **Torchvision Classifiers**

   - See: `PyTorch Vision Classification Models <https://docs.pytorch.org/vision/main/models.html#classification>`_

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

- **YOLOv3**: ``yolov3tu``, ``yolov3u``, ``yolov3sppu`` (via Ultralytics)
- **YOLOv5**: ``yolov5nu``, ``yolov5su``, ``yolov5mu``, ``yolov5lu``, ``yolov5xu`` and the ``6u`` variants (via Ultralytics)
- **YOLOv7**: All variants with pretrained weights
- **YOLOv8**: ``yolov8n``, ``yolov8s``, ``yolov8m``, ``yolov8l``, ``yolov8x`` (via Ultralytics)
- **YOLOv9**: All variants with pretrained weights
- **YOLOv10**: All variants with pretrained weights
- **YOLOv11**: ``yolov11n``, ``yolov11s``, ``yolov11m``, ``yolov11l``, ``yolov11x`` (via Ultralytics)
- **YOLOv12**: All variants with pretrained weights
- **YOLOv13**: ``yolov13n``, ``yolov13s``, ``yolov13l``, ``yolov13x``
- **YOLOv26**: ``yolov26n``, ``yolov26s``, ``yolov26m``, ``yolov26l``, ``yolov26x`` (via Ultralytics)
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

Depth Estimation Models
~~~~~~~~~~~~~~~~~~~~~~~

- **Depth-Anything-V1**: ``depth_anything_v1_small``, ``depth_anything_v1_base``, ``depth_anything_v1_large``
- **Depth-Anything-V2**: ``depth_anything_v2_small``, ``depth_anything_v2_base``, ``depth_anything_v2_large``
- **Depth-Anything-V3**: ``depth_anything_v3_mono_large``, ``depth_anything_v3_metric_large``

Hand-Object Interaction Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **HOI-DETR**: ``hoi_detr_vitl``
- **Hands23**: ``hands23_x101``

Example Usage
~~~~~~~~~~~~~

Download models using the CLI:

.. code-block:: console

    $ python -m trtutils download --model yolov8n --output yolov8n.onnx
    $ python -m trtutils download --model yolov11m --output yolov11m.onnx --imgsz 640 --opset 17
    $ python -m trtutils download --model yoloxs --output yoloxs.onnx --imgsz 640 --opset 17

For more information on the download command, see the :ref:`CLI Reference <cli>`.
