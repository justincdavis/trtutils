# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Engine-level tests for src/trtutils/models/promptable_segmenters/ -- MobileSAM, SAM and SAM 2.1."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import DATA_DIR
from trtutils.image import ImageEncoder, MaskDecoder
from trtutils.image._schema import MaskDecoderOutputSchema
from trtutils.models import (
    SAM,
    SAM2,
    MobileSAM,
    MobileSAMDecoder,
    MobileSAMEncoder,
    SAM2Decoder,
    SAM2Encoder,
    SAMDecoder,
    SAMEncoder,
)

# (composite, encoder, decoder, ONNX path with {} for the part)
FAMILIES = [
    pytest.param(
        (MobileSAM, MobileSAMEncoder, MobileSAMDecoder, "mobile_sam/mobile_sam_{}_1024.onnx"),
        id="mobile-sam",
    ),
    pytest.param((SAM, SAMEncoder, SAMDecoder, "sam/sam_vit_b_{}_1024.onnx"), id="sam-vit-b"),
    pytest.param(
        (SAM, SAMEncoder, SAMDecoder, "sam/sam_vit_l_{}_1024.onnx"),
        id="sam-vit-l",
        marks=pytest.mark.slow,
    ),
    *[
        pytest.param(
            (SAM2, SAM2Encoder, SAM2Decoder, f"sam2/sam2_1_hiera_{size}_{{}}_1024.onnx"),
            id=f"sam2.1-{size}",
            marks=marks,
        )
        for size, marks in (
            ("tiny", ()),
            ("small", ()),
            ("base_plus", ()),
            ("large", pytest.mark.slow),
        )
    ],
]
# the horse in data/horse.jpg, (x1, y1, x2, y2), and a point on its body
HORSE_BOX = np.array([65, 45, 392, 272], np.float32)
HORSE_POINT = np.array([[230, 110]], np.float32)


@pytest.fixture(scope="module", params=FAMILIES)
def family(request):
    """The wrapper classes and ONNX path pattern of one model."""
    return request.param


@pytest.fixture(scope="module")
def encoder_engine(build_model_engine, family):
    """Image encoder engine, FP16 as the build default."""
    _, encoder_cls, _, onnx = family
    return build_model_engine(encoder_cls, DATA_DIR / onnx.format("encoder"), tag="fp16")


@pytest.fixture(scope="module")
def decoder_engine(build_model_engine, family):
    """Mask decoder engine, at most 4 prompt sets of 8 points."""
    _, _, decoder_cls, onnx = family
    return build_model_engine(decoder_cls, DATA_DIR / onnx.format("decoder"), batch_size=4)


@pytest.fixture(scope="module")
def sam(family, encoder_engine, decoder_engine):
    """The composite on the CUDA preprocessor."""
    return family[0](encoder_engine, decoder_engine, warmup=False)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def _assert_masks_equal(got: list, ref: list) -> None:
    assert len(got) == len(ref)
    for (mask, score), (ref_mask, ref_score) in zip(got, ref):
        np.testing.assert_array_equal(mask, ref_mask)
        assert score == ref_score


@pytest.mark.parametrize(
    ("prompt", "count"),
    [
        pytest.param({"points": HORSE_POINT}, 1, id="point"),
        pytest.param({"points": HORSE_POINT, "labels": np.array([0.0])}, 1, id="background-point"),
        pytest.param({"boxes": HORSE_BOX}, 1, id="box"),
        pytest.param({"points": HORSE_POINT, "boxes": HORSE_BOX}, 1, id="point-and-box"),
        pytest.param({"boxes": np.stack([HORSE_BOX, HORSE_BOX + 5])}, 2, id="two-boxes"),
        pytest.param({"boxes": np.zeros((0, 4), np.float32)}, 0, id="no-boxes"),
    ],
)
@pytest.mark.cuda_graph
def test_predict_matches_end2end(sam, images, prompt, count) -> None:
    """set_image + predict and end2end give the same masks, one per prompt set at image size."""
    image = images["horse"].array
    sam.set_image(image)
    masks = sam.predict(**prompt)
    _assert_masks_equal(sam.end2end(image, **prompt), masks)
    assert len(masks) == count
    for mask, score in masks:
        assert mask.shape == image.shape[:2]
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)) <= {0, 1}
        assert isinstance(score, float)


@pytest.mark.correctness
@pytest.mark.parametrize("preprocessor", ["cpu", "cuda"])
@pytest.mark.parametrize("multimask", [True, False])
def test_box_mask_covers_horse(
    family, encoder_engine, decoder_engine, images, preprocessor, multimask
) -> None:
    """A box around the horse gives a confident mask that fills much of the box and stays in it."""
    sam = family[0](encoder_engine, decoder_engine, preprocessor=preprocessor, warmup=False)
    ((mask, score),) = sam.end2end(images["horse"].array, boxes=HORSE_BOX, multimask=multimask)
    x1, y1, x2, y2 = HORSE_BOX.astype(int)
    inside = mask[y1:y2, x1:x2].sum()
    assert score > 0.8
    assert inside / mask.sum() > 0.97
    assert inside / ((x2 - x1) * (y2 - y1)) > 0.3


@pytest.mark.cuda_graph
def test_results_stable_across_images_and_calls(sam, images) -> None:
    """Repeated predicts, other images in between, and encoder runs never change an image's masks."""
    horse, people = images["horse"].array, images["people"].array
    sam.set_image(horse)
    ref = sam.predict(boxes=HORSE_BOX)
    # cuda graphs capture on the second call at a shape, replay from the third
    for _ in range(3):
        _assert_masks_equal(sam.predict(boxes=HORSE_BOX), ref)
    # an encoder run outside the composite leaves its embeddings alone
    sam.encoder.encode(people)
    _assert_masks_equal(sam.predict(boxes=HORSE_BOX), ref)
    sam.set_image(people)
    ((other, _),) = sam.predict(boxes=HORSE_BOX)
    assert other.shape == people.shape[:2]
    sam.set_image(horse)
    _assert_masks_equal(sam.predict(boxes=HORSE_BOX), ref)


@pytest.mark.cuda_graph
def test_varying_point_counts(sam, images) -> None:
    """Prompts of 2 to 8 decoder points in any order give the same masks as when run alone."""
    sam.set_image(images["horse"].array)
    rng = np.random.default_rng(0)
    prompts = [
        {"points": HORSE_POINT + rng.uniform(-20, 20, (n, 2)).astype(np.float32)} for n in (1, 3, 7)
    ]
    prompts.append({"points": HORSE_POINT, "boxes": HORSE_BOX})
    refs = [sam.predict(**p) for p in prompts]
    for i in (3, 0, 2, 1, 0, 3, 2):
        _assert_masks_equal(sam.predict(**prompts[i]), refs[i])


def test_more_boxes_than_profile(sam, images) -> None:
    """More boxes than the profile's maximum run in chunks, in order, same as one by one."""
    sam.set_image(images["horse"].array)
    boxes = np.stack([HORSE_BOX + 4 * i for i in range(sam.decoder.max_prompts * 2 + 1)])
    masks = sam.predict(boxes=boxes, multimask=False)
    assert len(masks) == len(boxes)
    for box, (mask, score) in zip(boxes, masks):
        ((ref_mask, ref_score),) = sam.predict(boxes=box, multimask=False)
        assert _iou(mask.astype(bool), ref_mask.astype(bool)) > 0.99
        assert score == pytest.approx(ref_score, abs=1e-3)


def test_prompt_errors(family, sam, encoder_engine, decoder_engine, images) -> None:
    """Predicting without an image, or with more points than the profile, raises."""
    fresh = family[0](encoder_engine, decoder_engine, warmup=False)
    with pytest.raises(RuntimeError, match="set_image"):
        fresh.predict(boxes=HORSE_BOX)
    sam.set_image(images["horse"].array)
    points = np.repeat(HORSE_POINT, sam.decoder.max_points, axis=0)
    with pytest.raises(ValueError, match="at most"):
        sam.predict(points=points)


def test_schemas_and_wrong_engines(family, encoder_engine, decoder_engine) -> None:
    """Each part resolves its schema, the decoder reads the encoder outputs first, wrong parts raise."""
    _, encoder_cls, decoder_cls, _ = family
    encoder = encoder_cls(encoder_engine, warmup=False)
    decoder = decoder_cls(decoder_engine, warmup=False)
    features = encoder.output_schema.value
    assert decoder.input_schema.value[: len(features)] == features
    assert decoder.output_schema is MaskDecoderOutputSchema.SAM
    assert (decoder.max_prompts, decoder.max_points) == (4, 8)
    with pytest.raises(ValueError, match="match no"):
        MaskDecoder(encoder_engine, warmup=False)
    with pytest.raises(ValueError, match="image input"):
        ImageEncoder(decoder_engine, warmup=False)
