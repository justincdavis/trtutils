# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Classifier model implementations."""

from __future__ import annotations

from ._alexnet import AlexNet
from ._convnext import ConvNeXt
from ._densenet import DenseNet
from ._efficientnet import EfficientNet, EfficientNetV2
from ._googlenet import GoogLeNet
from ._inception import Inception
from ._maxvit import MaxViT
from ._mnasnet import MNASNet
from ._mobilenet import MobileNetV2, MobileNetV3
from ._regnet import RegNet
from ._resnet import ResNet, ResNeXt, WideResNet
from ._shufflenet import ShuffleNetV2
from ._squeezenet import SqueezeNet
from ._swin import SwinTransformer, SwinTransformerV2
from ._vgg import VGG
from ._vit import ViT

__all__ = [
    "VGG",
    "AlexNet",
    "ConvNeXt",
    "DenseNet",
    "EfficientNet",
    "EfficientNetV2",
    "GoogLeNet",
    "Inception",
    "MNASNet",
    "MaxViT",
    "MobileNetV2",
    "MobileNetV3",
    "RegNet",
    "ResNeXt",
    "ResNet",
    "ShuffleNetV2",
    "SqueezeNet",
    "SwinTransformer",
    "SwinTransformerV2",
    "ViT",
    "WideResNet",
]
