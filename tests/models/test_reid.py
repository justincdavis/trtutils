# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Engine-level tests for the re-ID wrappers in src/trtutils/models/reid/."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import pytest

from tests.conftest import DATA_DIR, PREPROCESSORS
from trtutils.image._schema import FeatureExtractorOutputSchema
from trtutils.image.preprocessors import crop_boxes
from trtutils.models import OSNet

_OSNET = DATA_DIR / "osnet"
_WIDTH = 512

# the three walkers in people.jpeg (x1, y1, x2, y2)
_PEOPLE = [(52, 21, 107, 168), (177, 15, 223, 171), (120, 35, 170, 169)]

_VARIANTS = [
    f"osnet_{width}_{dataset}"
    for dataset in ("msmt17", "market1501")
    for width in ("x0_25", "x0_5", "x0_75", "x1_0")
]
OSNETS = [
    pytest.param(_OSNET / f"{name}_256x128.onnx", id=name.replace("_", "-")) for name in _VARIANTS
]
# fully convolutional: a non-default (height, width) export builds and runs too
SIZED_OSNETS = [
    *OSNETS,
    pytest.param(_OSNET / "osnet_x0_25_msmt17_320x160.onnx", id="osnet-x0-25-msmt17-320x160"),
]


def _views(image, box) -> list[np.ndarray]:
    """The crop of a box, shifted, shrunk, grown, and mirrored."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    boxes = [
        box,
        (x1 + 0.06 * w, y1 + 0.04 * h, x2 + 0.06 * w, y2 + 0.04 * h),
        (x1 - 0.06 * w, y1 - 0.04 * h, x2 - 0.06 * w, y2 - 0.04 * h),
        (x1 + 0.05 * w, y1 + 0.05 * h, x2 - 0.05 * w, y2 - 0.05 * h),
        (x1 - 0.05 * w, y1 - 0.05 * h, x2 + 0.05 * w, y2 + 0.05 * h),
    ]
    crops = [np.ascontiguousarray(crop) for crop in crop_boxes(image, boxes)]
    return [*crops, np.ascontiguousarray(crops[0][:, ::-1])]


@pytest.mark.parametrize("onnx_path", SIZED_OSNETS)
@pytest.mark.parametrize(("preprocessor", "cpu_resize"), PREPROCESSORS)
def test_osnet_end2end(build_model_engine, images, onnx_path, preprocessor, cpu_resize) -> None:
    """A 512-d unit-norm embedding by default; run() + get_embeddings() match end2end()."""
    engine = build_model_engine(OSNet, onnx_path)
    model = OSNet(engine, preprocessor=preprocessor, cpu_resize=cpu_resize, warmup=False)
    assert model.output_schema is FeatureExtractorOutputSchema.EMBEDDING
    crop = np.ascontiguousarray(crop_boxes(images["people"].array, _PEOPLE[:1])[0])

    embedding = model.get_embeddings(model.run(crop))
    np.testing.assert_array_equal(embedding, model.end2end(crop))
    assert embedding.shape == (_WIDTH,)
    assert embedding.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(embedding), 1.0, rtol=1e-5)

    raw = model.end2end(crop, normalize=False)
    np.testing.assert_allclose(embedding, raw / np.linalg.norm(raw), rtol=1e-5, atol=1e-7)
    with pytest.raises(ValueError, match="has no feature map"):
        model.get_feature_maps(model.run(crop))


@pytest.mark.parametrize("onnx_path", OSNETS)
def test_osnet_end2end_boxes_chunks(build_model_engine, images, onnx_path) -> None:
    """More boxes than the dynamic batch run as a full and a short chunk; rows match their crops."""
    model = OSNet(build_model_engine(OSNet, onnx_path, batch_size=4), warmup=False)
    assert model.is_dynamic_batch is True
    image = images["people"].array
    boxes = [*_PEOPLE, (0, 0, 60, 120), (200, 60, 275, 183), (107, 75, 121, 114)]

    with patch.object(model, "end2end", wraps=model.end2end) as spy:
        rows = model.end2end_boxes(image, boxes)
    assert [len(call.args[0]) for call in spy.call_args_list] == [4, 2]
    assert rows.shape == (len(boxes), _WIDTH)
    assert rows.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(rows, axis=1), 1.0, rtol=1e-5)
    for row, crop in zip(rows, crop_boxes(image, boxes)):
        assert float(np.dot(row, model.end2end(np.ascontiguousarray(crop)))) > 0.9999


@pytest.mark.correctness
@pytest.mark.parametrize("onnx_path", OSNETS)
def test_osnet_same_person_closer(build_model_engine, images, onnx_path) -> None:
    """Jittered and mirrored crops of one person embed closer than crops of other people."""
    model = OSNet(build_model_engine(OSNet, onnx_path, batch_size=4), warmup=False)
    image = images["people"].array
    embeddings = [np.stack([model.end2end(view) for view in _views(image, box)]) for box in _PEOPLE]

    same = min(float((e @ e.T).min()) for e in embeddings)
    different = max(
        float((first @ second.T).max())
        for i, first in enumerate(embeddings)
        for second in embeddings[i + 1 :]
    )
    # measured on the 5080: same >= 0.91, different <= 0.73 across the variants
    assert same > 0.85
    assert different < 0.8


@pytest.mark.correctness
@pytest.mark.parametrize("onnx_path", OSNETS)
def test_osnet_matches_onnxruntime(build_model_engine, images, onnx_path) -> None:
    """The default build (FP16 where the platform has it) embeds like onnxruntime on the same input."""
    model = OSNet(
        build_model_engine(OSNet, onnx_path, batch_size=4), preprocessor="cpu", warmup=False
    )
    crops = [np.ascontiguousarray(c) for c in crop_boxes(images["people"].array, _PEOPLE)]

    tensor, _ratios, _padding = model.preprocess(crops, method="cpu")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    reference = session.run(["embedding"], {"input": tensor})[0]
    reference /= np.linalg.norm(reference, axis=1, keepdims=True)

    for row, ref in zip(model.end2end(crops), reference):
        assert float(np.dot(row, ref)) > 0.999
