# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/_image_model.py -- ImageModel preprocessing and engine helpers."""

from __future__ import annotations

from unittest import mock

import numpy as np
import nvtx
import pytest

from trtutils._flags import FLAGS
from trtutils._log import LOG
from trtutils.image import Detector, ImageModel, _image_model, preprocessors
from trtutils.image.preprocessors import CPUPreprocessor, CUDAPreprocessor


@pytest.mark.parametrize(
    ("cpu_resize", "atol"),
    [
        # the cuda resize kernels differ from cv2 by at most a pixel level
        pytest.param(None, 2 / 255, id="cuda"),
        pytest.param(True, 0.0, id="cuda-cpu-resize"),
    ],
)
def test_fp16_input_gpu_preprocessor_matches_cpu(
    yolov10_fp16_engine, images, cpu_resize, atol
) -> None:
    """A float16-input model keeps its GPU preprocessor and matches the CPU tensor and detections."""
    model = Detector(yolov10_fp16_engine, preprocessor="cuda", cpu_resize=cpu_resize, warmup=False)
    assert type(model.get_preprocessor()) is CUDAPreprocessor
    image = images["horse"].array

    tensor, ratios, padding = model.preprocess(image)
    expected, cpu_ratios, cpu_padding = model.preprocess(image, method="cpu")
    assert tensor.dtype == expected.dtype == np.float16
    assert (ratios, padding) == (cpu_ratios, cpu_padding)
    np.testing.assert_allclose(tensor, expected, rtol=0, atol=atol)

    bboxes, scores, class_ids = model.run(image)
    cpu_bboxes, cpu_scores, cpu_class_ids = model.run(
        expected, cpu_ratios, cpu_padding, preprocessed=True
    )
    np.testing.assert_allclose(bboxes, cpu_bboxes, atol=1.0)
    np.testing.assert_allclose(scores, cpu_scores, atol=0.01)
    np.testing.assert_array_equal(class_ids, cpu_class_ids)


def test_non_float_input_falls_back_to_cpu(uint8_nchw_engine) -> None:
    """A uint8 NCHW input, which the GPU preprocessors cannot write, warns and uses the CPU."""
    with mock.patch.object(LOG, "warning", wraps=LOG.warning) as warning:
        model = ImageModel(uint8_nchw_engine, preprocessor="cuda", warmup=False)
    assert type(model.get_preprocessor()) is CPUPreprocessor
    assert any("fallback to CPU" in call.args[0] for call in warning.call_args_list)


def _bare_model(
    cpu_resize: bool | None = None, dtype: type = np.float32, *, sst_in_graph: bool = False
) -> ImageModel:
    """An ImageModel with only the state get_preprocessor reads, without an engine."""
    model = ImageModel.__new__(ImageModel)
    model._tag = "test"
    model._nvtx_tags = {}
    model._device = None
    model._engine = mock.Mock()
    model._input_size = (640, 640)
    model._input_range = (0.0, 1.0)
    model._dtype = np.dtype(dtype)
    model._batch_size = 1
    model._resize_method = "letterbox"
    model._crop_pct = 0.875
    model._antialias = True
    model._mean = None
    model._std = None
    model._pagelocked_mem = True
    model._unified_mem = None
    model._sst_in_graph = sst_in_graph
    model._cpu_resize = cpu_resize
    model._preproc_cuda = None
    model._preproc_trt = None
    model._preproc_dla = {}
    return model


@pytest.mark.cpu
def test_hybrid_preprocessor_is_removed() -> None:
    """'hybrid' is no longer a preprocessing method, and the error lists the options."""
    with pytest.raises(ValueError, match=r"'hybrid'.*\['cpu', 'cuda', 'trt', 'dla0', 'dla1'\]"):
        _bare_model()._resolve_method("hybrid")
    assert not hasattr(preprocessors, "HybridPreprocessor")


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("method", "cls_name"),
    [("cuda", "CUDAPreprocessor"), ("trt", "TRTPreprocessor"), ("dla1", "DLAPreprocessor")],
)
def test_get_preprocessor_forwards_cpu_resize(monkeypatch, method, cls_name) -> None:
    """The model's cpu_resize reaches the preprocessor of every device method."""
    monkeypatch.setattr(FLAGS, "TRT_HAS_UINT8", True)
    cls = mock.Mock()
    monkeypatch.setattr(_image_model, cls_name, cls)
    model = _bare_model(cpu_resize=False)
    assert model.get_preprocessor(method) is cls.return_value
    assert cls.call_args.kwargs["cpu_resize"] is False
    assert cls.call_args.kwargs["antialias"] is True


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("method", "dtype", "sst_in_graph", "expected"),
    [
        pytest.param("cuda", np.float32, False, "cuda", id="cuda"),
        pytest.param("cuda", np.int8, False, "cpu", id="cuda-int8"),
        pytest.param("trt", np.uint8, True, "cuda", id="trt-sst-in-graph"),
        pytest.param("dla0", np.float16, False, "dla0", id="dla0"),
    ],
)
def test_resolve_method(method, dtype, sst_in_graph, expected) -> None:
    """Methods resolve with their fallbacks, an in-graph normalize to 'cuda' whatever the method."""
    model = _bare_model(None, dtype, sst_in_graph=sst_in_graph)
    assert model._resolve_method(method) == expected


def test_get_random_input_and_mock_run(yolov10_engine) -> None:
    """get_random_input() matches the engine input spec and mock_run() matches the output spec."""
    model = ImageModel(yolov10_engine, warmup=False)
    rand_input = model.get_random_input()
    assert [(list(a.shape), a.dtype) for a in rand_input] == model.engine.input_spec
    outputs = model.mock_run()
    assert [(list(o.shape), o.dtype) for o in outputs] == model.engine.output_spec


def test_init_failure_balances_nvtx_ranges(yolov10_engine, monkeypatch) -> None:
    """A constructor that raises still pops every NVTX range it pushed."""
    depth = []
    monkeypatch.setattr(FLAGS, "NVTX_ENABLED", True)
    monkeypatch.setattr(nvtx, "push_range", depth.append)
    monkeypatch.setattr(nvtx, "pop_range", depth.pop)
    monkeypatch.setattr(ImageModel, "_configure_model", mock.Mock(side_effect=ValueError("bad")))
    with pytest.raises(ValueError, match="bad"):
        ImageModel(yolov10_engine, warmup=False)
    assert depth == []
