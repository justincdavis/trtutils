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

11. **RT-DETRv4**

    - GitHub: `RT-DETRs/RT-DETRv4 <https://github.com/RT-DETRs/RT-DETRv4>`_

12. **D-FINE**

    - GitHub: `Peterande/D-FINE <https://github.com/Peterande/D-FINE>`_

13. **DEIM**

    - GitHub: `Intellindust-AI-Lab/DEIM <https://github.com/Intellindust-AI-Lab/DEIM>`_

14. **DEIMv2**

    - GitHub: `Intellindust-AI-Lab/DEIMv2 <https://github.com/Intellindust-AI-Lab/DEIMv2>`_

15. **RF-DETR**
    
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

1. **DINOv3** (``DINOv3``) - self-supervised ViT backbones (LVD-1689M weights), exported
   through timm with a class-token embedding and a dense patch feature map

   - GitHub: `facebookresearch/dinov3 <https://github.com/facebookresearch/dinov3>`_
   - Images are resized with ``linear``, which does not antialias. Embeddings agree across the
     trtutils preprocessors, but for heavily downscaled images they differ from embeddings
     computed with an antialiased resize such as upstream's transform (cosine similarity
     0.96 to 0.98 on average and as low as 0.90 on the test images). Embed queries and
     galleries with the same pipeline

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

1. **Ultralytics YOLO** - YOLOv8, YOLOv9, YOLOv11, and YOLOv26 segmentation heads

   - YOLOv8: ``yolov8n-seg``, ``yolov8s-seg``, ``yolov8m-seg``, ``yolov8l-seg``, ``yolov8x-seg``
   - YOLOv9: ``yolov9c-seg``, ``yolov9e-seg`` (the only sizes Ultralytics publishes)
   - YOLOv11: ``yolov11n-seg``, ``yolov11s-seg``, ``yolov11m-seg``, ``yolov11l-seg``, ``yolov11x-seg``
   - YOLOv26: ``yolov26n-seg``, ``yolov26s-seg``, ``yolov26m-seg``, ``yolov26l-seg``, ``yolov26x-seg``
   - GitHub: `ultralytics/ultralytics <https://github.com/ultralytics/ultralytics>`_
   - AGPL-3.0 and commercial licensed, be aware of license restrictions

2. **YOLOv12-seg** - the YOLOv12 authors' segmentation models (Ultralytics publishes none)

   - ``yolov12n-seg``, ``yolov12s-seg``, ``yolov12m-seg``, ``yolov12l-seg``, ``yolov12x-seg``
   - GitHub: `sunsmarterjie/yolov12 <https://github.com/sunsmarterjie/yolov12/tree/Seg>`_
   - AGPL-3.0 licensed, be aware of license restrictions

Semantic Segmentation
----------------------

:class:`~trtutils.image.SemanticSegmenter` returns one class map per image: an ``(H, W)``
uint8 array of class ids at the original resolution (so at most 256 classes). For one mask
per detected object, use the instance segmentation models above
(:class:`~trtutils.image.Segmenter`).

1. **DeepLabV3** (torchvision) - MobileNetV3-Large, ResNet50, and ResNet101 backbones

   - ``deeplabv3_mobilenet_v3_large``, ``deeplabv3_resnet50``, ``deeplabv3_resnet101``
   - 21 Pascal VOC classes (0 is background), trained on a COCO subset
   - Default input 520x520 (torchvision evaluates at a 520 short side) with letterbox
     resizing; ``DeepLabV3.build()`` runs the argmax in the engine (``argmax=False`` keeps
     the logits)
   - GitHub: `pytorch/vision <https://github.com/pytorch/vision>`_

Pose Estimation
----------------

1. **Ultralytics YOLO** - YOLOv8, YOLOv11, and YOLOv26 pose heads

   - YOLOv8: ``yolov8n-pose``, ``yolov8s-pose``, ``yolov8m-pose``, ``yolov8l-pose``, ``yolov8x-pose``
   - YOLOv11: ``yolov11n-pose``, ``yolov11s-pose``, ``yolov11m-pose``, ``yolov11l-pose``, ``yolov11x-pose``
   - YOLOv26: ``yolov26n-pose``, ``yolov26s-pose``, ``yolov26m-pose``, ``yolov26l-pose``, ``yolov26x-pose``
   - GitHub: `ultralytics/ultralytics <https://github.com/ultralytics/ultralytics>`_
   - AGPL-3.0 and commercial licensed, be aware of license restrictions

2. **RTMO** - one-stage COCO-17 body keypoints (``models.RTMO``): ``rtmo_s``, ``rtmo_m``,
   ``rtmo_l`` (640x640, trained on body7)

   - GitHub: `open-mmlab/mmpose (projects/rtmo) <https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo>`_
   - Downloads are OpenMMLab's ready ONNX (checked against a pinned sha256). The exporter flips
     the BGR input to RGB in the graph and pads the in-graph NMS output (IoU 0.65, at most 50
     people) to a static 50 rows, so engines allocate their outputs up front. The engine keeps
     the people scoring at least 0.1 (mmpose's ``score_thr``; OpenMMLab's export used 0.15) plus
     the next-best row: an NMS that selects nothing breaks TensorRT 8.6 on images without people.
     ``conf_thres`` (default 0.1) filters the rows
   - Preprocessing is mmpose's: a centred letterbox with gray (114) padding, 0-255 pixels,
     no mean / std. Postprocessing runs on the CPU and CUDA graphs default to off (the NMS
     inside the engine has data-dependent shapes)

Face Detection
~~~~~~~~~~~~~~

Face detectors run through ``PoseEstimator``: each result is a face box, its score and five
landmarks as ``(x, y, score)``, in the order right eye, left eye, nose tip, right mouth corner,
left mouth corner. The landmarks have no scores of their own, so their third value repeats the
face score.

1. **YuNet** - OpenCV Zoo's face detector (``models.YuNet``): ``yunet`` (640x640 by default,
   any size divisible by 32 through ``--imgsz``)

   - GitHub: `opencv/opencv_zoo (models/face_detection_yunet) <https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet>`_,
     trained with `ShiqiYu/libfacedetection.train <https://github.com/ShiqiYu/libfacedetection.train>`_
   - Downloads are the zoo's ready ONNX (``face_detection_yunet_2026may.onnx``, checked against a
     pinned sha256). The exporter flips the BGR input to RGB in the graph and fixes the input
     size, batch 1
   - Preprocessing is a centred letterbox to the engine size, 0-255 pixels, no mean / std.
     OpenCV's ``FaceDetectorYN`` runs on the image at its own size; letterboxing keeps the
     aspect ratio the faces were trained with. Postprocessing reproduces ``FaceDetectorYN``
     (score ``sqrt(cls * obj)``, per-stride decode, NMS) on the CPU, with its default
     thresholds: ``conf_thres=0.9``, ``nms_iou_thres=0.3``

Oriented Bounding Boxes
-------------------------

1. **Ultralytics YOLO** - YOLOv8, YOLOv11, and YOLOv26 OBB heads

   - YOLOv8: ``yolov8n-obb``, ``yolov8s-obb``, ``yolov8m-obb``, ``yolov8l-obb``, ``yolov8x-obb``
   - YOLOv11: ``yolov11n-obb``, ``yolov11s-obb``, ``yolov11m-obb``, ``yolov11l-obb``, ``yolov11x-obb``
   - YOLOv26: ``yolov26n-obb``, ``yolov26s-obb``, ``yolov26m-obb``, ``yolov26l-obb``, ``yolov26x-obb``
   - GitHub: `ultralytics/ultralytics <https://github.com/ultralytics/ultralytics>`_
   - AGPL-3.0 and commercial licensed, be aware of license restrictions

Note: YOLOv12 has no pose or OBB weights upstream, and its classification weights (published
by the YOLOv12 authors, not by Ultralytics) are not packaged yet, so it is absent from the pose,
OBB, and classification lists.

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
   can be processed without rebuilding. PP-OCRv4 and PP-OCRv5 (mobile and server) are
   supported. The bundled character dictionaries and the model weights are from PaddleOCR
   (Apache-2.0): ``ppocr_keys_v1.txt`` is the default and fits PP-OCRv4; PP-OCRv5 recognizers
   need ``character_dict=models.ocr.PPOCR_KEYS_V5`` (``ppocrv5_dict.txt``) on
   ``PPOCRRecognizer`` and ``PPOCR``.

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
- **RT-DETRv4**: ``rtdetrv4_s``, ``rtdetrv4_m``, ``rtdetrv4_l``, ``rtdetrv4_x``
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

Hand-Object Interaction Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **HOI-DETR**: ``hoi_detr_vitl``
- **Hands23**: ``hands23_x101``

OCR Models
~~~~~~~~~~

- **PP-OCR**: ``ppocrv4_det``, ``ppocrv4_rec`` (PP-OCRv4 Chinese/English, ready-made ONNX);
  ``ppocrv5_mobile_det``, ``ppocrv5_server_det``, ``ppocrv5_mobile_rec``,
  ``ppocrv5_server_rec`` (PP-OCRv5 simplified/traditional Chinese, English and
  Japanese; RapidOCR's ONNX conversions, checked against a pinned sha256). The server
  detector's neck is rescaled at download so it fits FP16, which needs ``onnx``
  (``trtutils[onnx]``)

Instance Segmentation Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **YOLOv8-seg**: ``yolov8n-seg``, ``yolov8s-seg``, ``yolov8m-seg``, ``yolov8l-seg``, ``yolov8x-seg`` (via Ultralytics)
- **YOLOv9-seg**: ``yolov9c-seg``, ``yolov9e-seg`` (via Ultralytics)
- **YOLOv12-seg**: ``yolov12n-seg``, ``yolov12s-seg``, ``yolov12m-seg``, ``yolov12l-seg``, ``yolov12x-seg``
- **YOLOv11-seg**: ``yolov11n-seg``, ``yolov11s-seg``, ``yolov11m-seg``, ``yolov11l-seg``, ``yolov11x-seg`` (via Ultralytics)
- **YOLOv26-seg**: ``yolov26n-seg``, ``yolov26s-seg``, ``yolov26m-seg``, ``yolov26l-seg``, ``yolov26x-seg`` (via Ultralytics)

Semantic Segmentation Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **DeepLabV3**: ``deeplabv3_mobilenet_v3_large``, ``deeplabv3_resnet50``, ``deeplabv3_resnet101`` (via torchvision)

Pose Estimation Models
~~~~~~~~~~~~~~~~~~~~~~~

- **YOLOv8-pose**: ``yolov8n-pose``, ``yolov8s-pose``, ``yolov8m-pose``, ``yolov8l-pose``, ``yolov8x-pose`` (via Ultralytics)
- **YOLOv11-pose**: ``yolov11n-pose``, ``yolov11s-pose``, ``yolov11m-pose``, ``yolov11l-pose``, ``yolov11x-pose`` (via Ultralytics)
- **YOLOv26-pose**: ``yolov26n-pose``, ``yolov26s-pose``, ``yolov26m-pose``, ``yolov26l-pose``, ``yolov26x-pose`` (via Ultralytics)
- **RTMO**: ``rtmo_s``, ``rtmo_m``, ``rtmo_l`` (OpenMMLab's ONNX)
- **YuNet** (face detection): ``yunet`` (OpenCV Zoo's ONNX)

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
       YOLOv9-seg; the ``u`` variants of YOLOv9, v10, v12; ``rtdetrv1_lu`` / ``rtdetrv1_xu``)
     - AGPL-3.0 (Ultralytics also sells an enterprise license)
   * - YOLOv7, YOLOv9
     - GPL-3.0
   * - YOLOv10, YOLOv12 (detection and seg), YOLOv13
     - AGPL-3.0
   * - YOLOX
     - Apache-2.0
   * - RT-DETRv1, RT-DETRv2, RT-DETRv3
     - Apache-2.0
   * - RT-DETRv4
     - Apache-2.0 (RT-DETRv4's license; the checkpoints, hosted on Google Drive, have no
       separate statement)
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
   * - DeepLabV3 (torchvision)
     - unknown (torchvision states no license for its weights)
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
   * - HOI-DETR
     - MIT
   * - Hands23
     - unknown (the code is MIT; the weights state no license)
   * - PP-OCR (v4, v5)
     - Apache-2.0
   * - RTMO
     - Apache-2.0 (mmpose's license; upstream states no separate license for the weights,
       which are trained on body7, seven public datasets with their own terms)
   * - YuNet
     - MIT (the license of the OpenCV Zoo model directory)

Example Usage
~~~~~~~~~~~~~

Download models using the CLI:

.. code-block:: console

    $ python -m trtutils download --model yolov8n --output yolov8n.onnx
    $ python -m trtutils download --model yolov11m --output yolov11m.onnx --imgsz 640 --opset 17
    $ python -m trtutils download --model yoloxs --output yoloxs.onnx --imgsz 640 --opset 17

For more information on the download command, see the :ref:`CLI Reference <cli>`.
