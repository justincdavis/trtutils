# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Postprocessors for images.

Functions
---------
:func:`get_classifications`
    Get the classifications from the output of a classification model.
:func:`get_depth_maps`
    Get the depth maps from the output of a depth estimation model.
:func:`postprocess_classifications`
    Postprocess the output of a classification model.
:func:`postprocess_depth`
    Postprocess the output of a depth estimation model.
:func:`get_detections`
    Get the detections from unified postprocessed outputs.
:func:`get_embeddings`
    Get the embeddings from the output of a feature extraction model.
:func:`get_feature_maps`
    Get the dense feature maps from the output of a feature extraction model.
:func:`postprocess_features`
    Postprocess the output of a feature extraction model.
:func:`get_text_boxes`
    Get the text boxes from the output of a text detection model.
:func:`get_texts`
    Get the texts from the output of a text recognition model.
:func:`get_interactions`
    Pair hands with objects from postprocessed hand-object interaction outputs.
:func:`postprocess_hand_interactions`
    Postprocess the output of a hand-object interaction model.
:func:`postprocess_yolov10`
    Postprocess the output of a YOLO-v10 model.
:func:`postprocess_yolo_seg`
    Postprocess the output of a YOLO segmentation model.
:func:`get_segmentations`
    Get the segmentations from postprocessed segmentation outputs.
:func:`postprocess_class_map`
    Postprocess the class map of a semantic segmentation model.
:func:`postprocess_semantic_logits`
    Postprocess the logits of a semantic segmentation model.
:func:`get_class_maps`
    Get the class maps from postprocessed semantic segmentation outputs.
:func:`postprocess_yolo_pose`
    Postprocess the output of a YOLO pose estimation model.
:func:`postprocess_rtmo_pose`
    Postprocess the output of an RTMO pose estimation model.
:func:`get_poses`
    Get the poses from postprocessed pose estimation outputs.
:func:`postprocess_yolo_obb`
    Postprocess the output of a YOLO oriented bounding box model.
:func:`get_obb_detections`
    Get the oriented bounding box detections from postprocessed outputs.
:func:`postprocess_rfdetr`
    Postprocess the output of a RF-DETR model.
:func:`postprocess_detr`
    Postprocess the output of a DETR-based model.
:func:`postprocess_detr_lbs`
    Postprocess the output of a DETR-based model with LBS output order.
:func:`postprocess_rtdetrv3`
    Postprocess the output of an RT-DETR v3 model.
:func:`postprocess_efficient_nms`
    Postprocess the output of an EfficientNMS model.
:func:`postprocess_text_detection`
    Postprocess the output of a DB text detection model.
:func:`postprocess_text_recognition`
    Postprocess the output of a CTC text recognition model.
:func:`sort_text_boxes`
    Sort text boxes in reading order.

"""

from __future__ import annotations

from trtutils.image.types import HandInteraction, OBBDetection, Pose, Segmentation

from ._cpu_classifier import get_classifications, postprocess_classifications
from ._cpu_depth import get_depth_maps, postprocess_depth
from ._cpu_detection import (
    get_detections,
    postprocess_detr,
    postprocess_detr_lbs,
    postprocess_efficient_nms,
    postprocess_rfdetr,
    postprocess_rtdetrv3,
    postprocess_yolov10,
)
from ._cpu_features import get_embeddings, get_feature_maps, postprocess_features
from ._cpu_hand_interaction import get_interactions, postprocess_hand_interactions
from ._cpu_obb import get_obb_detections, postprocess_yolo_obb
from ._cpu_ocr import (
    get_text_boxes,
    get_texts,
    postprocess_text_detection,
    postprocess_text_recognition,
    sort_text_boxes,
)
from ._cpu_pose import get_poses, postprocess_rtmo_pose, postprocess_yolo_pose
from ._cpu_segmentation import get_segmentations, postprocess_yolo_seg
from ._cpu_semantic import get_class_maps, postprocess_class_map, postprocess_semantic_logits

__all__ = [
    "HandInteraction",
    "OBBDetection",
    "Pose",
    "Segmentation",
    "get_class_maps",
    "get_classifications",
    "get_depth_maps",
    "get_detections",
    "get_embeddings",
    "get_feature_maps",
    "get_interactions",
    "get_obb_detections",
    "get_poses",
    "get_segmentations",
    "get_text_boxes",
    "get_texts",
    "postprocess_class_map",
    "postprocess_classifications",
    "postprocess_depth",
    "postprocess_detr",
    "postprocess_detr_lbs",
    "postprocess_efficient_nms",
    "postprocess_features",
    "postprocess_hand_interactions",
    "postprocess_rfdetr",
    "postprocess_rtdetrv3",
    "postprocess_rtmo_pose",
    "postprocess_semantic_logits",
    "postprocess_text_detection",
    "postprocess_text_recognition",
    "postprocess_yolo_obb",
    "postprocess_yolo_pose",
    "postprocess_yolo_seg",
    "postprocess_yolov10",
    "sort_text_boxes",
]
