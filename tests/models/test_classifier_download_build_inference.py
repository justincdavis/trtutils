# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="misc"
from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import pytest

from trtutils.models import (
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

from .paths import HORSE_IMAGE_PATH

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def _temporary_dir() -> Iterator[Path]:
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path)


# Each classifier config: (ModelClass, model_name, imgsz)
# imgsz is None to use class default (224 for most, 299 for Inception)
CLASSIFIER_CONFIGS = [
    (AlexNet, "alexnet", None),
    (ConvNeXt, "convnext_tiny", None),
    (DenseNet, "densenet121", None),
    (EfficientNet, "efficientnet_b0", None),
    (EfficientNetV2, "efficientnet_v2_s", None),
    (GoogLeNet, "googlenet", None),
    (Inception, "inception_v3", None),
    (MaxViT, "maxvit_t", None),
    (MNASNet, "mnasnet0_5", None),
    (MobileNetV2, "mobilenet_v2", None),
    (MobileNetV3, "mobilenet_v3_small", None),
    (RegNet, "regnet_y_400mf", None),
    (ResNet, "resnet18", None),
    (ResNeXt, "resnext50_32x4d", None),
    (ShuffleNetV2, "shufflenet_v2_x0_5", None),
    (SqueezeNet, "squeezenet1_0", None),
    (SwinTransformer, "swin_t", None),
    (SwinTransformerV2, "swin_v2_t", None),
    (VGG, "vgg11", None),
    (ViT, "vit_b_16", None),
    (WideResNet, "wide_resnet50_2", None),
]


@pytest.mark.parametrize(
    ("model_class", "model_name", "imgsz"),
    CLASSIFIER_CONFIGS,
    ids=[cfg[1] for cfg in CLASSIFIER_CONFIGS],
)
def test_classifier_download_build_inference(
    model_class: type, model_name: str, imgsz: int | None
) -> None:
    """
    Test the complete classifier workflow: download model, build engine, run inference.

    For each supported classifier model, this test:
    1. Downloads the model to ONNX via the class static method
    2. Builds a TensorRT engine
    3. Runs inference on the horse test image
    4. Asserts that classification outputs are produced
    """
    with _temporary_dir() as temp_dir:
        onnx_path = temp_dir / f"{model_name}.onnx"
        engine_path = temp_dir / f"{model_name}.engine"

        # Download the model to ONNX
        model_class.download(  # type: ignore[attr-defined]
            model=model_name,
            output=onnx_path,
            imgsz=imgsz,
            accept=True,
        )
        assert onnx_path.exists(), f"ONNX file was not created: {onnx_path}"

        # Build the TensorRT engine
        model_class.build(  # type: ignore[attr-defined]
            onnx=onnx_path,
            output=engine_path,
            imgsz=imgsz,
            opt_level=1,
        )
        assert engine_path.exists(), f"Engine file was not created: {engine_path}"

        # Load the model and run inference
        classifier = model_class(
            engine_path,
            warmup=False,
        )

        image = cv2.imread(HORSE_IMAGE_PATH)
        assert image is not None, f"Failed to read image: {HORSE_IMAGE_PATH}"

        classifications = classifier.end2end([image])

        # Assert classifications were produced
        assert len(classifications) == 1, "Expected classifications for 1 image"
        assert len(classifications[0]) >= 1, (
            f"Expected at least 1 classification, got {len(classifications[0])}"
        )

        # Each classification should be (class_id, confidence)
        class_id, confidence = classifications[0][0]
        assert isinstance(class_id, int), f"Expected int class_id, got {type(class_id)}"
        assert isinstance(confidence, float), f"Expected float confidence, got {type(confidence)}"
        assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} out of range [0, 1]"

        # Verify inference correctness: horse.jpg should be classified as a horse
        # ImageNet class 339 = "sorrel" (a reddish-brown horse, matches the test image)
        horse_class_id = 339
        top5_ids = [cid for cid, _ in classifications[0][:5]]
        assert horse_class_id in top5_ids, (
            f"Expected sorrel (class {horse_class_id}) in top-5 predictions, got {top5_ids}"
        )

        del classifier
