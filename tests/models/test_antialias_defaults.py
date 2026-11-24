# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/models/ -- which model wrappers antialias downscales by default."""

from __future__ import annotations

import inspect

import pytest

import trtutils.models as models
from trtutils.image import (
    Classifier,
    DepthEstimator,
    Detector,
    FeatureExtractor,
    HandInteractionDetector,
    OBBDetector,
    PoseEstimator,
    Segmenter,
    SemanticSegmenter,
)
from trtutils.image._image_model import ImageModel

pytestmark = pytest.mark.cpu

# upstream preprocessing resizes these with PIL / torchvision antialiasing; changing this set
# changes their default outputs, so it must be deliberate
ANTIALIASED = {
    # torchvision semantic segmenters
    "DeepLabV3",
    # torchvision classifiers
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
    "VGG",
    "ViT",
    "WideResNet",
    # feature extractors whose timm transform resizes PIL images bicubic
    "DINOv3",
    # timm classifiers
    "DeiT",
    "DeiT3",
    # DETR detectors whose upstream resizes PIL images with torchvision
    "DEIM",
    "DEIMv2",
    "DFINE",
    "RFDETR",
    "RTDETRv1",
    "RTDETRv2",
}
# OCR preprocesses with its own bit-exact OpenCV pipeline and has no antialias option
NO_OPTION = {"PPOCR", "PPOCRDetector", "PPOCRRecognizer"}


def _default(cls: type) -> object:
    return inspect.signature(cls.__init__).parameters["antialias"].default


@pytest.mark.parametrize("name", sorted(set(models.__all__) - NO_OPTION))
def test_wrapper_antialias_default(name) -> None:
    """Wrappers of antialiasing upstreams default antialias on, all others off."""
    cls = getattr(models, name)
    assert issubclass(cls, ImageModel)
    assert _default(cls) is (name in ANTIALIASED)


@pytest.mark.parametrize(
    "cls",
    [
        pytest.param(cls, id=cls.__name__)
        for cls in (
            ImageModel,
            Detector,
            Classifier,
            DepthEstimator,
            FeatureExtractor,
            HandInteractionDetector,
            Segmenter,
            PoseEstimator,
            OBBDetector,
            SemanticSegmenter,
        )
    ],
)
def test_task_antialias_default(cls) -> None:
    """Task models default antialias off."""
    assert _default(cls) is False


def test_antialiased_names_exist() -> None:
    """Every pinned name is a model wrapper."""
    assert set(models.__all__) >= ANTIALIASED
