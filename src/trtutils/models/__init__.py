# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Implementations of various deep learning models.

Classes
-------
:class:`YOLO`
    Alias for the Detector class with args preset for YOLO.
:class:`YOLOv3`
    Alias for the YOLO class with args preset for YOLOv3.
:class:`YOLOv5`
    Alias for the YOLO class with args preset for YOLOv5.
:class:`YOLOv7`
    Alias for the YOLO class with args preset for YOLOv7.
:class:`YOLOv8`
    Alias for the YOLO class with args preset for YOLOv8.
:class:`YOLOv9`
    Alias for the YOLO class with args preset for YOLOv9.
:class:`YOLOv10`
    Alias for the YOLO class with args preset for YOLOv10.
:class:`YOLOv11`
    Alias for the YOLO class with args preset for YOLOv11.
:class:`YOLOv12`
    Alias for the YOLO class with args preset for YOLOv12.
:class:`YOLOv13`
    Alias for the YOLO class with args preset for YOLOv13.
:class:`YOLOX`
    Alias for the YOLO class with args preset for YOLOX.
:class:`DETR`
    Alias for the Detector class with args preset for DETR.
:class:`RTDETRv1`
    Alias for the DETR class with args preset for RT-DETRv1.
:class:`RTDETRv2`
    Alias for the DETR class with args preset for RT-DETRv2.
:class:`RTDETRv3`
    Alias for the DETR class with args preset for RT-DETRv3.
:class:`DFINE`
    Alias for the DETR class with args preset for D-FINE.
:class:`DEIM`
    Alias for the DETR class with args preset for DEIM.
:class:`DEIMv2`
    Alias for the DETR class with args preset for DEIMv2.
:class:`RFDETR`
    Alias for the DETR class with args preset for RF-DETR.
:class:`AlexNet`
    Alias for the Classifier class with args preset for AlexNet.
:class:`ConvNeXt`
    Alias for the Classifier class with args preset for ConvNeXt.
:class:`DenseNet`
    Alias for the Classifier class with args preset for DenseNet.
:class:`EfficientNet`
    Alias for the Classifier class with args preset for EfficientNet.
:class:`EfficientNetV2`
    Alias for the Classifier class with args preset for EfficientNet V2.
:class:`GoogLeNet`
    Alias for the Classifier class with args preset for GoogLeNet.
:class:`Inception`
    Alias for the Classifier class with args preset for Inception V3.
:class:`MaxViT`
    Alias for the Classifier class with args preset for MaxViT.
:class:`MNASNet`
    Alias for the Classifier class with args preset for MNASNet.
:class:`MobileNetV2`
    Alias for the Classifier class with args preset for MobileNet V2.
:class:`MobileNetV3`
    Alias for the Classifier class with args preset for MobileNet V3.
:class:`RegNet`
    Alias for the Classifier class with args preset for RegNet.
:class:`ResNet`
    Alias for the Classifier class with args preset for ResNet.
:class:`ResNeXt`
    Alias for the Classifier class with args preset for ResNeXt.
:class:`ShuffleNetV2`
    Alias for the Classifier class with args preset for ShuffleNet V2.
:class:`SqueezeNet`
    Alias for the Classifier class with args preset for SqueezeNet.
:class:`SwinTransformer`
    Alias for the Classifier class with args preset for Swin Transformer.
:class:`SwinTransformerV2`
    Alias for the Classifier class with args preset for Swin Transformer V2.
:class:`VGG`
    Alias for the Classifier class with args preset for VGG.
:class:`ViT`
    Alias for the Classifier class with args preset for ViT.
:class:`WideResNet`
    Alias for the Classifier class with args preset for Wide ResNet.
:class:`DepthAnythingV2`
    Alias for the DepthEstimator class with args preset for Depth-Anything-V2.

"""

from __future__ import annotations

from .classifiers import (
    VGG,
    AlexNet,
    ConvNeXt,
    DenseNet,
    EfficientNet,
    EfficientNetV2,
    GoogLeNet,
    Inception,
    MaxViT,
    MNASNet,
    MobileNetV2,
    MobileNetV3,
    RegNet,
    ResNet,
    ResNeXt,
    ShuffleNetV2,
    SqueezeNet,
    SwinTransformer,
    SwinTransformerV2,
    ViT,
    WideResNet,
)
from .depth_estimators import DepthAnythingV2
from .detectors import (
    DEIM,
    DETR,
    DFINE,
    RFDETR,
    YOLO,
    YOLOX,
    DEIMv2,
    RTDETRv1,
    RTDETRv2,
    RTDETRv3,
    YOLOv3,
    YOLOv5,
    YOLOv7,
    YOLOv8,
    YOLOv9,
    YOLOv10,
    YOLOv11,
    YOLOv12,
    YOLOv13,
    YOLOv26,
)

__all__ = [
    "DEIM",
    "DETR",
    "DFINE",
    "RFDETR",
    "VGG",
    "YOLO",
    "YOLOX",
    "AlexNet",
    "ConvNeXt",
    "DEIMv2",
    "DenseNet",
    "DepthAnythingV2",
    "EfficientNet",
    "EfficientNetV2",
    "GoogLeNet",
    "Inception",
    "MNASNet",
    "MaxViT",
    "MobileNetV2",
    "MobileNetV3",
    "RTDETRv1",
    "RTDETRv2",
    "RTDETRv3",
    "RegNet",
    "ResNeXt",
    "ResNet",
    "ShuffleNetV2",
    "SqueezeNet",
    "SwinTransformer",
    "SwinTransformerV2",
    "ViT",
    "WideResNet",
    "YOLOv3",
    "YOLOv5",
    "YOLOv7",
    "YOLOv8",
    "YOLOv9",
    "YOLOv10",
    "YOLOv11",
    "YOLOv12",
    "YOLOv13",
    "YOLOv26",
]
