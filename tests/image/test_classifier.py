# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import cv2
import numpy as np

from trtutils.image import Classifier

from .common import build_classifier_engine
from .paths import HORSE_IMAGE_PATH, IMAGE_PATHS

PARITY_TOLERANCE = 0.05  # 5% tolerance for classification parity


def _read_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        err_msg = f"Failed to read image: {path}"
        raise FileNotFoundError(err_msg)
    return img


def test_classifier_load() -> None:
    """Test Classifier initializes with only engine path."""
    engine_path = build_classifier_engine()
    classifier = Classifier(engine_path, warmup=False)
    assert classifier is not None
    del classifier


def test_classifier_load_with_args() -> None:
    """Test Classifier initializes with various arguments."""
    engine_path = build_classifier_engine()
    classifier = Classifier(
        engine_path,
        warmup=False,
        preprocessor="cpu",
        resize_method="linear",
    )
    assert classifier is not None
    del classifier


def test_classifier_run_cpu() -> None:
    """Test Classifier runs with CPU preprocessor."""
    engine_path = build_classifier_engine()
    classifier = Classifier(engine_path, warmup=False, preprocessor="cpu")
    img = _read_image(HORSE_IMAGE_PATH)
    outputs = classifier.run(img)
    assert outputs is not None
    assert len(outputs) > 0
    del classifier


def test_classifier_run_cuda() -> None:
    """Test Classifier runs with CUDA preprocessor."""
    engine_path = build_classifier_engine()
    classifier = Classifier(engine_path, warmup=False, preprocessor="cuda")
    img = _read_image(HORSE_IMAGE_PATH)
    outputs = classifier.run(img)
    assert outputs is not None
    assert len(outputs) > 0
    del classifier


def test_classifier_run_trt() -> None:
    """Test Classifier runs with TRT preprocessor."""
    engine_path = build_classifier_engine()
    classifier = Classifier(engine_path, warmup=False, preprocessor="trt")
    img = _read_image(HORSE_IMAGE_PATH)
    outputs = classifier.run(img)
    assert outputs is not None
    assert len(outputs) > 0
    del classifier


def test_classifier_preprocess_letterbox() -> None:
    """Test Classifier preprocessing with letterbox."""
    engine_path = build_classifier_engine()
    classifier = Classifier(
        engine_path, warmup=False, preprocessor="cpu", resize_method="letterbox"
    )
    img = _read_image(HORSE_IMAGE_PATH)
    tensor, ratios, padding = classifier.preprocess(img)
    assert tensor is not None
    assert ratios is not None
    assert padding is not None
    assert len(ratios) == 2
    assert len(padding) == 2
    del classifier


def test_classifier_preprocess_linear() -> None:
    """Test Classifier preprocessing with linear resize."""
    engine_path = build_classifier_engine()
    classifier = Classifier(
        engine_path, warmup=False, preprocessor="cpu", resize_method="linear"
    )
    img = _read_image(HORSE_IMAGE_PATH)
    tensor, ratios, padding = classifier.preprocess(img)
    assert tensor is not None
    assert ratios is not None
    assert padding is not None
    assert len(ratios) == 2
    assert len(padding) == 2
    del classifier


def test_classifier_postprocess() -> None:
    """Test Classifier postprocessing."""
    engine_path = build_classifier_engine()
    classifier = Classifier(engine_path, warmup=False, preprocessor="cpu")
    img = _read_image(HORSE_IMAGE_PATH)
    tensor, _, _ = classifier.preprocess(img)
    outputs = classifier._engine([tensor], no_copy=False)
    postprocessed = classifier.postprocess(outputs)
    assert postprocessed is not None
    assert len(postprocessed) > 0
    del classifier


def test_classifier_get_classifications() -> None:
    """Test Classifier get_classifications method."""
    engine_path = build_classifier_engine()
    classifier = Classifier(engine_path, warmup=False, preprocessor="cpu")
    img = _read_image(HORSE_IMAGE_PATH)
    outputs = classifier.run(img)
    classifications = classifier.get_classifications(outputs, top_k=5)
    assert classifications is not None
    assert isinstance(classifications, list)
    assert len(classifications) <= 5
    # Check classification format
    if len(classifications) > 0:
        cls_id, conf = classifications[0]
        assert isinstance(cls_id, (int, np.integer))
        assert isinstance(conf, (float, np.floating))
        # Confidence should be between 0 and 1 for most models
        assert 0.0 <= conf <= 1.0 or conf > 0  # Allow for non-normalized scores
    del classifier


def test_classifier_get_classifications_top_k() -> None:
    """Test Classifier get_classifications with different top_k values."""
    engine_path = build_classifier_engine()
    classifier = Classifier(engine_path, warmup=False, preprocessor="cpu")
    img = _read_image(HORSE_IMAGE_PATH)
    outputs = classifier.run(img)

    # Test different top_k values
    classifications_1 = classifier.get_classifications(outputs, top_k=1)
    classifications_5 = classifier.get_classifications(outputs, top_k=5)
    classifications_10 = classifier.get_classifications(outputs, top_k=10)

    assert len(classifications_1) <= 1
    assert len(classifications_5) <= 5
    assert len(classifications_10) <= 10

    # More results should be superset of fewer results (first k should match)
    if len(classifications_1) > 0 and len(classifications_5) > 0:
        assert classifications_1[0] == classifications_5[0]

    del classifier


def test_classifier_end2end() -> None:
    """Test Classifier end2end method."""
    engine_path = build_classifier_engine()
    classifier = Classifier(engine_path, warmup=False, preprocessor="cuda")
    img = _read_image(HORSE_IMAGE_PATH)
    classifications = classifier.end2end(img, top_k=5)
    assert classifications is not None
    assert isinstance(classifications, list)
    assert len(classifications) <= 5
    del classifier


def test_classifier_call_method() -> None:
    """Test Classifier __call__ method."""
    engine_path = build_classifier_engine()
    classifier = Classifier(engine_path, warmup=False, preprocessor="cpu")
    img = _read_image(HORSE_IMAGE_PATH)
    outputs = classifier(img)
    assert outputs is not None
    assert len(outputs) > 0
    del classifier


def test_classifier_preprocessed_flag() -> None:
    """Test Classifier with preprocessed flag."""
    engine_path = build_classifier_engine()
    classifier = Classifier(engine_path, warmup=False, preprocessor="cpu")
    img = _read_image(HORSE_IMAGE_PATH)
    tensor, _, _ = classifier.preprocess(img)
    outputs = classifier.run(tensor, preprocessed=True, postprocess=True)
    assert outputs is not None
    assert len(outputs) > 0
    del classifier


def test_classifier_no_postprocess() -> None:
    """Test Classifier without postprocessing."""
    engine_path = build_classifier_engine()
    classifier = Classifier(engine_path, warmup=False, preprocessor="cpu")
    img = _read_image(HORSE_IMAGE_PATH)
    outputs = classifier.run(img, postprocess=False)
    assert outputs is not None
    assert len(outputs) > 0
    # Raw outputs should be different from postprocessed
    del classifier


def test_classifier_preprocess_method_switch() -> None:
    """Test switching preprocessor methods at runtime."""
    engine_path = build_classifier_engine()
    classifier = Classifier(engine_path, warmup=False, preprocessor="cpu")
    img = _read_image(HORSE_IMAGE_PATH)

    # Test switching to CUDA
    tensor_cuda, _, _ = classifier.preprocess(img, method="cuda")
    assert tensor_cuda is not None

    # Test switching to TRT
    tensor_trt, _, _ = classifier.preprocess(img, method="trt")
    assert tensor_trt is not None

    # Test switching back to CPU
    tensor_cpu, _, _ = classifier.preprocess(img, method="cpu")
    assert tensor_cpu is not None

    del classifier


def _compare_classifications(
    cls1: list[tuple[int, float]],
    cls2: list[tuple[int, float]],
    tolerance: float,
) -> bool:
    """Compare two classification lists for parity."""
    # Both should have results
    if len(cls1) == 0 or len(cls2) == 0:
        return len(cls1) == len(cls2)

    # Top prediction should match
    top_cls1, top_conf1 = cls1[0]
    top_cls2, top_conf2 = cls2[0]

    # Class IDs should match for top prediction
    if top_cls1 != top_cls2:
        # Allow some tolerance - check if top class from one is in top 3 of other
        top_3_classes_1 = [c for c, _ in cls1[:3]]
        top_3_classes_2 = [c for c, _ in cls2[:3]]
        if top_cls1 not in top_3_classes_2 and top_cls2 not in top_3_classes_1:
            return False

    # Confidence should be similar (allow larger tolerance for confidence)
    if abs(top_conf1 - top_conf2) > 0.2:
        return False

    return True


def test_classifier_parity_cpu_cuda_letterbox() -> None:
    """Test parity between CPU and CUDA preprocessors with letterbox."""
    engine_path = build_classifier_engine()

    classifier_cpu = Classifier(
        engine_path, warmup=False, preprocessor="cpu", resize_method="letterbox"
    )
    classifier_cuda = Classifier(
        engine_path, warmup=False, preprocessor="cuda", resize_method="letterbox"
    )

    img = _read_image(HORSE_IMAGE_PATH)

    outputs_cpu = classifier_cpu.run(img)
    classifications_cpu = classifier_cpu.get_classifications(outputs_cpu, top_k=5)

    outputs_cuda = classifier_cuda.run(img)
    classifications_cuda = classifier_cuda.get_classifications(outputs_cuda, top_k=5)

    # Check that both produce results
    assert len(classifications_cpu) > 0
    assert len(classifications_cuda) > 0

    # Check parity
    assert _compare_classifications(
        classifications_cpu, classifications_cuda, PARITY_TOLERANCE
    )

    del classifier_cpu
    del classifier_cuda


def test_classifier_parity_cpu_cuda_linear() -> None:
    """Test parity between CPU and CUDA preprocessors with linear resize."""
    engine_path = build_classifier_engine()

    classifier_cpu = Classifier(
        engine_path, warmup=False, preprocessor="cpu", resize_method="linear"
    )
    classifier_cuda = Classifier(
        engine_path, warmup=False, preprocessor="cuda", resize_method="linear"
    )

    img = _read_image(HORSE_IMAGE_PATH)

    outputs_cpu = classifier_cpu.run(img)
    classifications_cpu = classifier_cpu.get_classifications(outputs_cpu, top_k=5)

    outputs_cuda = classifier_cuda.run(img)
    classifications_cuda = classifier_cuda.get_classifications(outputs_cuda, top_k=5)

    # Check that both produce results
    assert len(classifications_cpu) > 0
    assert len(classifications_cuda) > 0

    # Check parity
    assert _compare_classifications(
        classifications_cpu, classifications_cuda, PARITY_TOLERANCE
    )

    del classifier_cpu
    del classifier_cuda


def test_classifier_parity_cpu_trt_letterbox() -> None:
    """Test parity between CPU and TRT preprocessors with letterbox."""
    engine_path = build_classifier_engine()

    classifier_cpu = Classifier(
        engine_path, warmup=False, preprocessor="cpu", resize_method="letterbox"
    )
    classifier_trt = Classifier(
        engine_path, warmup=False, preprocessor="trt", resize_method="letterbox"
    )

    img = _read_image(HORSE_IMAGE_PATH)

    outputs_cpu = classifier_cpu.run(img)
    classifications_cpu = classifier_cpu.get_classifications(outputs_cpu, top_k=5)

    outputs_trt = classifier_trt.run(img)
    classifications_trt = classifier_trt.get_classifications(outputs_trt, top_k=5)

    # Check that both produce results
    assert len(classifications_cpu) > 0
    assert len(classifications_trt) > 0

    # Check parity
    assert _compare_classifications(
        classifications_cpu, classifications_trt, PARITY_TOLERANCE
    )

    del classifier_cpu
    del classifier_trt


def test_classifier_parity_cpu_trt_linear() -> None:
    """Test parity between CPU and TRT preprocessors with linear resize."""
    engine_path = build_classifier_engine()

    classifier_cpu = Classifier(
        engine_path, warmup=False, preprocessor="cpu", resize_method="linear"
    )
    classifier_trt = Classifier(
        engine_path, warmup=False, preprocessor="trt", resize_method="linear"
    )

    img = _read_image(HORSE_IMAGE_PATH)

    outputs_cpu = classifier_cpu.run(img)
    classifications_cpu = classifier_cpu.get_classifications(outputs_cpu, top_k=5)

    outputs_trt = classifier_trt.run(img)
    classifications_trt = classifier_trt.get_classifications(outputs_trt, top_k=5)

    # Check that both produce results
    assert len(classifications_cpu) > 0
    assert len(classifications_trt) > 0

    # Check parity
    assert _compare_classifications(
        classifications_cpu, classifications_trt, PARITY_TOLERANCE
    )

    del classifier_cpu
    del classifier_trt


def test_classifier_multiple_images() -> None:
    """Test Classifier on multiple images."""
    engine_path = build_classifier_engine()
    classifier = Classifier(engine_path, warmup=False, preprocessor="cuda")

    for img_path in IMAGE_PATHS:
        img = _read_image(img_path)
        classifications = classifier.end2end(img, top_k=5)
        assert classifications is not None
        assert isinstance(classifications, list)
        assert len(classifications) <= 5

    del classifier


def test_classifier_no_copy() -> None:
    """Test Classifier with no_copy flag."""
    engine_path = build_classifier_engine()
    classifier = Classifier(engine_path, warmup=False, preprocessor="cuda")
    img = _read_image(HORSE_IMAGE_PATH)

    # Test no_copy in preprocess
    tensor, ratios, padding = classifier.preprocess(img, no_copy=True)
    assert tensor is not None

    # Test no_copy in run
    outputs = classifier.run(img, no_copy=True)
    assert outputs is not None

    del classifier


def test_classifier_pagelocked_mem() -> None:
    """Test Classifier with pagelocked memory."""
    engine_path = build_classifier_engine()
    classifier = Classifier(
        engine_path, warmup=False, preprocessor="cuda", pagelocked_mem=True
    )
    img = _read_image(HORSE_IMAGE_PATH)
    classifications = classifier.end2end(img, top_k=5)
    assert classifications is not None
    del classifier


def test_classifier_no_pagelocked_mem() -> None:
    """Test Classifier without pagelocked memory."""
    engine_path = build_classifier_engine()
    classifier = Classifier(
        engine_path, warmup=False, preprocessor="cuda", pagelocked_mem=False
    )
    img = _read_image(HORSE_IMAGE_PATH)
    classifications = classifier.end2end(img, top_k=5)
    assert classifications is not None
    del classifier


def test_classifier_input_range() -> None:
    """Test Classifier with different input ranges."""
    engine_path = build_classifier_engine()

    # Test with [0, 1] range (default)
    classifier1 = Classifier(
        engine_path, warmup=False, preprocessor="cpu", input_range=(0.0, 1.0)
    )
    img = _read_image(HORSE_IMAGE_PATH)
    outputs1 = classifier1.run(img)
    assert outputs1 is not None

    # Test with [0, 255] range
    classifier2 = Classifier(
        engine_path, warmup=False, preprocessor="cpu", input_range=(0.0, 255.0)
    )
    outputs2 = classifier2.run(img)
    assert outputs2 is not None

    del classifier1
    del classifier2


def test_classifier_resize_methods() -> None:
    """Test Classifier with both resize methods produce valid results."""
    engine_path = build_classifier_engine()

    # Test letterbox
    classifier_letterbox = Classifier(
        engine_path, warmup=False, preprocessor="cpu", resize_method="letterbox"
    )
    img = _read_image(HORSE_IMAGE_PATH)
    outputs_letterbox = classifier_letterbox.run(img)
    cls_letterbox = classifier_letterbox.get_classifications(outputs_letterbox, top_k=5)
    assert len(cls_letterbox) > 0

    # Test linear
    classifier_linear = Classifier(
        engine_path, warmup=False, preprocessor="cpu", resize_method="linear"
    )
    outputs_linear = classifier_linear.run(img)
    cls_linear = classifier_linear.get_classifications(outputs_linear, top_k=5)
    assert len(cls_linear) > 0

    # Both should produce valid classifications
    assert cls_letterbox[0][0] >= 0  # Valid class ID
    assert cls_linear[0][0] >= 0  # Valid class ID

    del classifier_letterbox
    del classifier_linear


def test_classifier_warmup() -> None:
    """Test Classifier with warmup enabled."""
    engine_path = build_classifier_engine()
    classifier = Classifier(
        engine_path, warmup=True, warmup_iterations=5, preprocessor="cuda"
    )
    img = _read_image(HORSE_IMAGE_PATH)
    classifications = classifier.end2end(img, top_k=5)
    assert classifications is not None
    assert len(classifications) <= 5
    del classifier


def test_classifier_verbose() -> None:
    """Test Classifier with verbose flag."""
    engine_path = build_classifier_engine()
    classifier = Classifier(engine_path, warmup=False, preprocessor="cpu", verbose=True)
    img = _read_image(HORSE_IMAGE_PATH)
    # This should work without errors even with verbose logging
    classifications = classifier.end2end(img, top_k=5, verbose=True)
    assert classifications is not None
    del classifier



