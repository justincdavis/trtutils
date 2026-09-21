# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/_detector.py -- Detector inference on a YOLOv10 engine."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

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


@pytest.mark.parametrize("preprocessor", ["cuda", "trt", "cpu"])
def test_detector_end2end_resolution_switch(yolov10_engine, images, preprocessor) -> None:
    """end2end() survives resolution changes: a graph cache hit/miss, not a locked-dims error."""
    model = Detector(yolov10_engine, preprocessor=preprocessor, cuda_graph=True, warmup=False)
    horse = images["horse"].array
    sizes = [None, (1280, 720), (640, 480), None]  # None -> native size

    for size in sizes:
        image = horse if size is None else cv2.resize(horse, size)
        result = model.end2end(image)

        fresh = Detector(yolov10_engine, preprocessor=preprocessor, cuda_graph=True, warmup=False)
        expected = fresh.end2end(image)
        assert result == expected

    # single-image batch size and the preprocessed-output pointer never
    # change across resolutions, so only one graph should ever be cached
    assert len(model._e2e_graphs) == 1


@pytest.mark.parametrize("cuda_graph", [True, False])
def test_detector_static_engine_rejects_batch_mismatch(yolov10_engine, images, cuda_graph) -> None:
    """A static (batch-1) engine rejects a batch of 2 on end2end() and run(), graphed or not."""
    # cpu preprocessor: no fixed batch cap of its own, so the RuntimeError
    # raised is unambiguously _validate_batch_size's, not the TRT/CUDA
    # preprocessor's separate (and unrelated) configured-batch-size guard.
    model = Detector(yolov10_engine, preprocessor="cpu", cuda_graph=cuda_graph, warmup=False)
    batch = [images["horse"].array, images["horse"].array]

    with pytest.raises(RuntimeError):
        model.end2end(batch)
    with pytest.raises(RuntimeError):
        model.run(batch)


@pytest.mark.parametrize("preprocessor", ["cuda", "trt", "cpu"])
@pytest.mark.parametrize("cuda_graph", [True, False])
def test_detector_dynamic_batch_sweep(
    yolov10_dynamic_engine, yolov10_engine, images, preprocessor, cuda_graph
) -> None:
    """A dynamic-batch engine handles a variable batch sweep; matches a fresh static-b1 detector."""
    model = Detector(
        yolov10_dynamic_engine, preprocessor=preprocessor, cuda_graph=cuda_graph, warmup=False
    )
    reference = Detector(
        yolov10_engine, preprocessor=preprocessor, cuda_graph=cuda_graph, warmup=False
    )

    pool = [img.array for img in images.values()]

    for batch_size in (8, 2, 8, 4, 1, 8):
        batch = [pool[i % len(pool)] for i in range(batch_size)]
        outputs = model.end2end(batch)
        assert len(outputs) == batch_size

        for image, dets in zip(batch, outputs):
            ref_dets = reference.end2end(image)
            assert len(dets) == len(ref_dets)
            for (bbox, score, cls_id), (ref_bbox, ref_score, ref_cls_id) in zip(dets, ref_dets):
                # separate TensorRT builds (dynamic vs static profile) pick
                # different kernels, and the fp16 TRT preprocessor adds its
                # own rounding, so allow a one-pixel box edge difference
                assert all(abs(a - b) <= 1 for a, b in zip(bbox, ref_bbox))
                assert cls_id == ref_cls_id
                assert abs(score - ref_score) < 1e-2


def test_detector_run_direct_gpu_path_matches_host(yolov10_engine, images) -> None:
    """run() via the direct-GPU path (cuda preprocessor + cuda_graph) matches the CPU host path."""
    horse = images["horse"].array
    direct = Detector(yolov10_engine, preprocessor="cuda", cuda_graph=True, warmup=False)
    host = Detector(yolov10_engine, preprocessor="cpu", cuda_graph=False, warmup=False)

    direct_bboxes, direct_scores, direct_cls = direct.run(horse)
    host_bboxes, host_scores, host_cls = host.run(horse)

    # cuda and cpu resize kernels differ at the sub-pixel level, so bboxes
    # rescaled through them match closely but not bit-for-bit
    np.testing.assert_allclose(direct_bboxes, host_bboxes, atol=0.1)
    np.testing.assert_allclose(direct_scores, host_scores, atol=1e-3)
    np.testing.assert_array_equal(direct_cls, host_cls)


def test_detector_run_direct_gpu_path_no_copy(yolov10_engine, images) -> None:
    """run(postprocess=False) copies out of engine memory unless no_copy=True is passed."""
    horse = images["horse"].array
    model = Detector(yolov10_engine, preprocessor="cuda", cuda_graph=True, warmup=False)

    raw_copy = model.run(horse, postprocess=False)
    engine_alloc = model.engine._outputs[0].host_allocation
    assert not np.shares_memory(raw_copy[0], engine_alloc)

    raw_view = model.run(horse, postprocess=False, no_copy=True)
    assert np.shares_memory(raw_view[0], engine_alloc)


def _assert_dets_close(dets_a, dets_b) -> None:
    """Compare two detection lists: boxes within 1 px, scores within 1e-3, classes exact."""
    assert len(dets_a) == len(dets_b)
    for (box_a, score_a, cls_a), (box_b, score_b, cls_b) in zip(dets_a, dets_b):
        assert all(abs(a - b) <= 1 for a, b in zip(box_a, box_b))
        assert abs(score_a - score_b) < 1e-3
        assert cls_a == cls_b


@pytest.mark.parametrize("preprocessor", ["cuda", "trt"])
def test_detector_postprocessor_cuda_matches_cpu(yolov10_engine, images, preprocessor) -> None:
    """postprocessor='cuda' end2end detections match postprocessor='cpu' for YOLO_V10."""
    image = images["horse"].array
    cpu_model = Detector(
        yolov10_engine,
        preprocessor=preprocessor,
        cuda_graph=True,
        postprocessor="cpu",
        warmup=False,
    )
    cuda_model = Detector(
        yolov10_engine,
        preprocessor=preprocessor,
        cuda_graph=True,
        postprocessor="cuda",
        warmup=False,
    )
    assert cuda_model._gpu_postprocess

    cpu_dets = cpu_model.end2end(image)
    cuda_dets = cuda_model.end2end(image)
    _assert_dets_close(cpu_dets, cuda_dets)


def test_detector_postprocessor_cuda_dynamic_batch(yolov10_dynamic_engine, images) -> None:
    """
    postprocessor='cuda' matches 'cpu' across a batch sweep on a dynamic engine.

    Batch 4 is submitted twice with a batch-1 call in between, so the
    identity ratios/padding staged for the graphed rescale kernel must be
    re-uploaded on every call rather than reused from the first capture.
    """
    cpu_model = Detector(
        yolov10_dynamic_engine,
        preprocessor="cuda",
        cuda_graph=True,
        postprocessor="cpu",
        warmup=False,
    )
    cuda_model = Detector(
        yolov10_dynamic_engine,
        preprocessor="cuda",
        cuda_graph=True,
        postprocessor="cuda",
        warmup=False,
    )
    pool = [img.array for img in images.values()]

    for batch_size in (4, 1, 4):
        batch = [pool[i % len(pool)] for i in range(batch_size)]
        cpu_dets = cpu_model.end2end(batch)
        cuda_dets = cuda_model.end2end(batch)
        assert len(cuda_dets) == batch_size
        for cpu_d, cuda_d in zip(cpu_dets, cuda_dets):
            _assert_dets_close(cpu_d, cuda_d)


def test_detector_postprocessor_cuda_no_graph_matches_cpu(yolov10_engine, images) -> None:
    """With cuda_graph=False the postprocess hooks never fire, so 'cuda' behaves like 'cpu'."""
    image = images["horse"].array
    cpu_model = Detector(
        yolov10_engine,
        preprocessor="cuda",
        cuda_graph=False,
        postprocessor="cpu",
        warmup=False,
    )
    cuda_model = Detector(
        yolov10_engine,
        preprocessor="cuda",
        cuda_graph=False,
        postprocessor="cuda",
        warmup=False,
    )
    # the kernel/binding are still set up; cuda_graph=False just means the
    # hooks that would launch/stage them are never reached
    assert cuda_model._gpu_postprocess

    cpu_dets = cpu_model.end2end(image)
    cuda_dets = cuda_model.end2end(image)
    assert cpu_dets == cuda_dets


def test_detector_unknown_postprocessor_raises(yolov10_engine) -> None:
    """An unrecognized postprocessor string raises ValueError."""
    with pytest.raises(ValueError, match="Unknown postprocessor"):
        Detector(yolov10_engine, postprocessor="bogus", warmup=False)


@pytest.mark.parametrize("preprocessor", ["cuda", "trt"])
def test_detector_postprocessor_cuda_matches_cpu_efficient_nms(
    yolov8n_effnms_engine, images, preprocessor
) -> None:
    """postprocessor='cuda' matches 'cpu' end2end detections for an EfficientNMS-schema engine."""
    image = images["horse"].array
    cpu_model = Detector(
        yolov8n_effnms_engine,
        preprocessor=preprocessor,
        cuda_graph=True,
        postprocessor="cpu",
        warmup=False,
    )
    cuda_model = Detector(
        yolov8n_effnms_engine,
        preprocessor=preprocessor,
        cuda_graph=True,
        postprocessor="cuda",
        warmup=False,
    )
    assert cuda_model._gpu_postprocess

    cpu_dets = cpu_model.end2end(image)
    cuda_dets = cuda_model.end2end(image)
    _assert_dets_close(cpu_dets, cuda_dets)
