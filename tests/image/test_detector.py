# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/_detector.py -- Detector inference on a YOLOv10 engine."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from trtutils import Buffer, MemoryLocation
from trtutils.image import Detector


@pytest.mark.parametrize("preprocessor", ["cpu", "cuda", "trt"])
def test_detector_end2end(yolov10_engine, images, preprocessor) -> None:
    """run(), get_detections(), and end2end() agree, honor conf_thres, and find the ground truth."""
    model = Detector(yolov10_engine, preprocessor=preprocessor, warmup=False)
    image = images["horse"]
    height, width = image.array.shape[:2]

    raw = model.run(image.array, postprocess=False)
    assert [(list(o.shape), o.dtype) for o in raw] == model.engine.output_spec

    postprocessed = model.run(image.array)
    bboxes, scores, class_ids = postprocessed
    assert bboxes.shape == (len(scores), 4)
    assert class_ids.shape == scores.shape

    via_run = model.get_detections(postprocessed)
    via_e2e = model.end2end(image.array)
    assert via_run == via_e2e

    assert len(via_run) >= image.gt_det_min
    assert set(image.gt_det_classes) <= {cls_id for _bbox, _score, cls_id in via_run}
    for (x1, y1, x2, y2), score, _cls_id in via_run:
        assert 0 <= x1 <= x2 <= width
        assert 0 <= y1 <= y2 <= height
        assert 0.0 <= score <= 1.0

    strict = model.get_detections(postprocessed, conf_thres=0.5)
    assert len(strict) <= len(via_run)
    assert all(score >= 0.5 for _bbox, score, _cls_id in strict)


@pytest.mark.parametrize("preprocessor", ["cpu", "cuda", "trt"])
def test_detector_end2end_resolution_switch(yolov10_engine, images, preprocessor) -> None:
    """end2end() handles a new input resolution on every call."""
    model = Detector(yolov10_engine, preprocessor=preprocessor, warmup=False)
    horse = images["horse"].array
    for size in [None, (1280, 720), (640, 480), None]:
        image = horse if size is None else cv2.resize(horse, size)
        fresh = Detector(yolov10_engine, preprocessor=preprocessor, warmup=False)
        assert model.end2end(image) == fresh.end2end(image)


@pytest.mark.parametrize("cuda_graph", [True, False])
def test_detector_static_engine_rejects_batch_mismatch(yolov10_engine, images, cuda_graph) -> None:
    """A batch-1 engine rejects a batch of 2 on end2end() and run()."""
    model = Detector(yolov10_engine, preprocessor="cpu", cuda_graph=cuda_graph, warmup=False)
    batch = [images["horse"].array] * 2
    with pytest.raises((RuntimeError, ValueError)):
        model.end2end(batch)
    with pytest.raises((RuntimeError, ValueError)):
        model.run(batch)


@pytest.mark.parametrize("preprocessor", ["cpu", "cuda", "trt"])
@pytest.mark.parametrize("cuda_graph", [True, False])
def test_detector_dynamic_batch_sweep(
    yolov10_dynamic_engine, yolov10_engine, images, preprocessor, cuda_graph
) -> None:
    """A dynamic-batch engine runs a varying batch and matches a batch-1 engine per image."""
    kw = {"preprocessor": preprocessor, "cuda_graph": cuda_graph, "warmup": False}
    model = Detector(yolov10_dynamic_engine, **kw)
    reference = Detector(yolov10_engine, **kw)
    pool = [img.array for img in images.values()]
    for batch_size in (8, 2, 8, 4, 1, 8):
        batch = [pool[i % len(pool)] for i in range(batch_size)]
        outputs = model.end2end(batch)
        assert len(outputs) == batch_size
        for image, dets in zip(batch, outputs):
            ref_dets = reference.end2end(image)
            assert len(dets) == len(ref_dets)
            # separate builds pick different kernels, so allow a pixel on box edges
            for (bbox, score, cls_id), (ref_bbox, ref_score, ref_cls) in zip(dets, ref_dets):
                assert all(abs(a - b) <= 1 for a, b in zip(bbox, ref_bbox))
                assert cls_id == ref_cls
                assert abs(score - ref_score) < 1e-2


def test_detector_run_gpu_matches_cpu(yolov10_engine, images) -> None:
    """run() with a GPU preprocessor matches the CPU preprocessor path."""
    horse = images["horse"].array
    gpu = Detector(yolov10_engine, preprocessor="cuda", warmup=False).run(horse)
    cpu = Detector(yolov10_engine, preprocessor="cpu", warmup=False).run(horse)
    np.testing.assert_allclose(gpu[0], cpu[0], atol=0.1)
    np.testing.assert_allclose(gpu[1], cpu[1], atol=1e-3)
    np.testing.assert_array_equal(gpu[2], cpu[2])


@pytest.mark.parametrize("preprocessor", ["cuda", "trt", "cpu"])
@pytest.mark.parametrize("cuda_graph", [True, False])
def test_detector_device_buffer_input(yolov10_engine, images, preprocessor, cuda_graph) -> None:
    """Device Buffer images give the same results as arrays from end2end(), run() and preprocess()."""
    model = Detector(yolov10_engine, preprocessor=preprocessor, cuda_graph=cuda_graph, warmup=False)
    horse = images["horse"].array
    small = cv2.resize(horse, (640, 480))
    device = [Buffer.from_array(img, MemoryLocation.DEVICE) for img in (horse, small)]

    expected = model.end2end(horse)
    assert model.end2end(device[0]) == expected
    assert model.get_detections(model.run(device[0])) == expected

    tensor, ratios, padding = model.preprocess(horse)
    device_tensor, device_ratios, device_padding = model.preprocess(device[0])
    np.testing.assert_array_equal(device_tensor, tensor)
    assert (device_ratios, device_padding) == (ratios, padding)

    # an already preprocessed tensor on the device is bound as is
    preprocessed = Buffer.from_array(tensor, MemoryLocation.DEVICE)
    outputs = model.run(preprocessed, ratios=ratios[0], padding=padding[0], preprocessed=True)
    assert model.get_detections(outputs) == expected


def test_detector_mixed_buffer_batch(yolov10_dynamic_engine, images) -> None:
    """A dynamic-batch engine takes arrays and device Buffers of mixed resolutions in one batch."""
    model = Detector(yolov10_dynamic_engine, preprocessor="cuda", warmup=False)
    pool = [img.array for img in images.values()]
    batch = [
        Buffer.from_array(img, MemoryLocation.DEVICE) if i % 2 else img for i, img in enumerate(pool)
    ]
    assert model.end2end(batch) == model.end2end(pool)
