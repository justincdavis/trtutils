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

Example Usage
~~~~~~~~~~~~~

Download models using the CLI:

.. code-block:: console

    $ python -m trtutils download --model yolov8n --output yolov8n.onnx
    $ python -m trtutils download --model yolov11m --output yolov11m.onnx --imgsz 640 --opset 17
    $ python -m trtutils download --model yoloxs --output yoloxs.onnx --imgsz 640 --opset 17

For more information on the download command, see the :ref:`CLI Reference <cli>`.
