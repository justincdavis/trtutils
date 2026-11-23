.. _models:

Supported Models

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

Feature Extraction
------------------

Every feature extractor also embeds the boxes of a frame with ``end2end_boxes(image, boxes)``,
one ``(N, D)`` row per box (e.g. detections for tracking or re-ID), whatever the engine batch.

1. **DINOv3** (``DINOv3``) - self-supervised ViT backbones (LVD-1689M weights), exported
   through timm with a class-token embedding and a dense patch feature map

   - GitHub: `facebookresearch/dinov3 <https://github.com/facebookresearch/dinov3>`_
   - Images are resized with ``linear``, which does not antialias. Embeddings agree across the
     trtutils preprocessors, but for heavily downscaled images they differ from embeddings
     computed with an antialiased resize such as upstream's transform (cosine similarity
     0.96 to 0.98 on average and as low as 0.90 on the test images). Embed queries and
     galleries with the same pipeline

Person Re-Identification
------------------------

Re-ID models are ``FeatureExtractor`` models: ``end2end_boxes(frame, boxes)`` embeds the person
detections of a frame, and a tracker matches them across frames by cosine similarity.

1. **OSNet** (``OSNet``) - Omni-Scale Network person re-ID from torchreid, 512-d embeddings
   from a ``(height, width) = (256, 128)`` crop, L2 normalized by default

   - Widths ``x0_25``, ``x0_5``, ``x0_75``, ``x1_0``, trained on MSMT17 (``osnet_<width>_msmt17``)
     or Market-1501 (``osnet_<width>_market1501``)
   - GitHub: `KaiyangZhou/deep-person-reid <https://github.com/KaiyangZhou/deep-person-reid>`_
   - FP16 builds keep the depthwise convolutions in FP32: on TensorRT 10.15 many FP16 builds
     with a fixed batch otherwise pick depthwise kernels that make every embedding wrong
   - Engines are built with a dynamic batch (1 to ``batch_size``); build with a batch that
     fits your detections per frame, e.g. ``OSNet.build(onnx, engine, batch_size=32)``
   - Fully convolutional: other ``(height, width)`` sizes can be downloaded
     (``--imgsz 320 160``)

Classification
--------------

1. **Torchvision Classifiers**

   - See: `PyTorch Vision Classification Models <https://docs.pytorch.org/vision/main/models.html#classification>`_

2. **DeiT** (``DeiT``) - ImageNet-1k classifiers, exported through timm

   - ``deit_tiny``, ``deit_small``, ``deit_base``, ``deit_base_hr``
   - Distilled: ``deit_tiny_distilled``, ``deit_small_distilled``, ``deit_base_distilled``,
     ``deit_base_distilled_hr`` (class and distillation heads averaged, one output)
   - GitHub: `facebookresearch/deit <https://github.com/facebookresearch/deit>`_
   - Apache-2.0 licensed

3. **DeiT III** (``DeiT3``) - ImageNet-1k classifiers, exported through timm

   - ``deit3_small``, ``deit3_medium``, ``deit3_base``, ``deit3_large``, ``deit3_huge``,
     ``deit3_small_hr``, ``deit3_base_hr``, ``deit3_large_hr``
   - ImageNet-22k pretrained: the same names with ``_in22k`` (e.g. ``deit3_base_in22k``,
     ``deit3_base_in22k_hr``)
   - GitHub: `facebookresearch/deit <https://github.com/facebookresearch/deit>`_
   - Apache-2.0 licensed

   DeiT and DeiT III download at any ``imgsz`` divisible by the patch size (16, or 14 for
   ``deit3_huge``); timm resamples the position embeddings. ``_hr`` models are fine-tuned at
   384 and default to it, the others to 224. Both wrappers preprocess with ``center_crop`` and
   ``crop_pct=0.9``; the ``_hr`` and ``_in22k`` weights were evaluated with ``crop_pct=1.0``.

4. **Ultralytics YOLO** - YOLOv8, YOLOv11, and YOLOv26 classification heads

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

Note: YOLOv26 segmentation, pose, and OBB heads are exported raw (``end2end=False``): the
one-to-many head decoded with NMS in trtutils. Ultralytics' default ``.pt`` predict uses the
NMS-free one-to-one head, so results can differ slightly from it (they match ultralytics run
on the same raw export).

Hand-Object Interaction
------------------------

1. **HOI-DETR**

   - GitHub: `AhmadDarKhalil/HOI-DETR <https://github.com/AhmadDarKhalil/HOI-DETR>`_

2. **Hands23**

   - GitHub: `EvaCheng-cty/hands23_detector <https://github.com/EvaCheng-cty/hands23_detector>`_
   - Downloads drop the ``NonZero`` ops automatically (about 5-9% faster ``end2end``); re-download
     older Hands23 ONNX files. Assumes at least 1000 valid RPN proposals per image (1700+ seen
     at 800 px).

Text Detection and Recognition (OCR)
------------------------------------

1. **PaddleOCR (PP-OCR)** - DB text detection, CTC text recognition, and the combined
   two-stage pipeline (``models.PPOCRDetector``, ``models.PPOCRRecognizer``, ``models.PPOCR``).
   Engines are built with dynamic input shapes, so any image size and text line width
   can be processed without rebuilding. The bundled default character dictionary
   (``ppocr_keys_v1.txt``) and the model weights are from PaddleOCR (Apache-2.0).

   - GitHub: `PaddlePaddle/PaddleOCR <https://github.com/PaddlePaddle/PaddleOCR>`_

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

Feature Extraction Models
~~~~~~~~~~~~~~~~~~~~~~~~~

- **DINOv3**: ``dinov3_vits16``, ``dinov3_vits16plus``, ``dinov3_vitb16`` (via timm; default size
  256, any multiple of 16; batch-dynamic by default)

Person Re-Identification Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **OSNet**: ``osnet_x0_25_msmt17``, ``osnet_x0_5_msmt17``, ``osnet_x0_75_msmt17``,
  ``osnet_x1_0_msmt17``, ``osnet_x0_25_market1501``, ``osnet_x0_5_market1501``,
  ``osnet_x0_75_market1501``, ``osnet_x1_0_market1501`` (torchreid model zoo; default size
  256x128 as ``(height, width)``; batch-dynamic by default)

Hand-Object Interaction Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **HOI-DETR**: ``hoi_detr_vitl``
- **Hands23**: ``hands23_x101``

OCR Models
~~~~~~~~~~

- **PP-OCR**: ``ppocrv4_det``, ``ppocrv4_rec`` (PP-OCRv4 Chinese/English, ready-made ONNX)

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

Model Licenses
~~~~~~~~~~~~~~

Downloaded weights keep their upstream license; trtutils' MIT license does not cover them.
Every download logs the model's license (``download(..., no_warn=True)`` hides it), with a louder
notice for copyleft and non-commercial licenses. ``unknown`` means upstream states no license
for the weights: check the source before using them.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Family
     - License
   * - Ultralytics YOLO (YOLOv3, v5, v8, v11, v26 and their seg / cls / pose / obb heads;
       the ``u`` variants of YOLOv9, v10, v12; ``rtdetrv1_lu`` / ``rtdetrv1_xu``)
     - AGPL-3.0 (Ultralytics also sells an enterprise license)
   * - YOLOv7, YOLOv9
     - GPL-3.0
   * - YOLOv10, YOLOv12, YOLOv13
     - AGPL-3.0
   * - YOLOX
     - Apache-2.0
   * - RT-DETRv1, RT-DETRv2, RT-DETRv3
     - Apache-2.0
   * - D-FINE
     - Apache-2.0 (upstream notes that Objects365-pretrained checkpoints such as
       ``*_obj2coco`` may carry the dataset's terms)
   * - DEIM
     - Apache-2.0
   * - DEIMv2
     - DEIMv2 License: weights for non-commercial use only (upstream relicensed; it was
       Apache-2.0 at the commit the exporter pins)
   * - RF-DETR (``rfdetr_n``, ``rfdetr_s``, ``rfdetr_m``)
     - Apache-2.0
   * - Torchvision classifiers
     - unknown (torchvision states no license for its weights), except ``regnet_y_128gf``
       and ``vit_h_14`` (SWAG weights): CC-BY-NC-4.0
   * - DeiT, DeiT III (timm)
     - Apache-2.0
   * - Depth-Anything-V1
     - Apache-2.0
   * - Depth-Anything-V2
     - ``small``: Apache-2.0; ``base``, ``large``: CC-BY-NC-4.0
   * - Depth-Anything-V3 (``mono_large``, ``metric_large``)
     - Apache-2.0
   * - DINOv3 (timm mirrors of the LVD-1689M ViT-S, S+, B)
     - DINOv3 License: Meta's custom license, commercial use allowed; read its terms
       (the upstream Meta repositories are gated, the timm mirrors are not)
   * - OSNet (torchreid model zoo)
     - MIT (torchreid's license; the weights have no separate statement). They were trained
       on the MSMT17 and Market-1501 research datasets, whose own terms apply
   * - HOI-DETR
     - MIT
   * - Hands23
     - unknown (the code is MIT; the weights state no license)
   * - PP-OCR
     - Apache-2.0

Example Usage
~~~~~~~~~~~~~

Download models using the CLI:

.. code-block:: console

    $ python -m trtutils download --model yolov8n --output yolov8n.onnx
    $ python -m trtutils download --model yolov11m --output yolov11m.onnx --imgsz 640 --opset 17
    $ python -m trtutils download --model yoloxs --output yoloxs.onnx --imgsz 640 --opset 17

For more information on the download command, see the :ref:`CLI Reference <cli>`.
