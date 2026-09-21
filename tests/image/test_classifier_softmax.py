# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for the classification postprocessor softmax flag."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils.image.postprocessors import get_classifications, postprocess_classifications

pytestmark = pytest.mark.cpu

_LOGITS = np.array([[1.0, 2.0, 3.0, 0.5]], dtype=np.float32)


def test_softmax_true_produces_probabilities() -> None:
    """softmax=True converts logits into a distribution summing to 1.0."""
    outputs = [_LOGITS.copy()]
    result = postprocess_classifications(outputs, softmax=True)

    probs = result[0][0]
    assert probs.shape == _LOGITS.shape
    assert float(np.sum(probs)) == pytest.approx(1.0, abs=1e-5)
    # softmax should not be the identity on non-uniform logits
    assert not np.allclose(probs, _LOGITS)


def test_softmax_false_passes_through_untouched() -> None:
    """softmax=False leaves the raw (already-softmaxed) values unchanged."""
    already_softmaxed = np.array([[0.05, 0.1, 0.8, 0.05]], dtype=np.float32)
    outputs = [already_softmaxed.copy()]
    result = postprocess_classifications(outputs, softmax=False)

    np.testing.assert_array_equal(result[0][0], already_softmaxed)


def test_topk_ordering_identical_with_or_without_softmax() -> None:
    """Softmax is monotonic, so top-k label ordering must match either way."""
    outputs_true = [_LOGITS.copy()]
    outputs_false = [_LOGITS.copy()]

    post_true = postprocess_classifications(outputs_true, softmax=True)
    post_false = postprocess_classifications(outputs_false, softmax=False)

    top_true = get_classifications(post_true, top_k=4)[0]
    top_false = get_classifications(post_false, top_k=4)[0]

    labels_true = [idx for idx, _ in top_true]
    labels_false = [idx for idx, _ in top_false]
    assert labels_true == labels_false


def test_double_softmax_changes_confidences() -> None:
    """Softmaxing twice must differ from a single softmax (the bug this flag fixes)."""
    single = postprocess_classifications([_LOGITS.copy()], softmax=True)
    single_probs = single[0][0]

    # apply softmax a second time on top of the already-softmaxed probabilities
    double = postprocess_classifications([single_probs.copy()], softmax=True)
    double_probs = double[0][0]

    assert not np.allclose(single_probs, double_probs)


def test_no_copy_false_returns_copies() -> None:
    """no_copy=False (default) must not let mutations of the result affect the input."""
    original = _LOGITS.copy()
    outputs = [original.copy()]
    result = postprocess_classifications(outputs, softmax=True, no_copy=False)

    result[0][0][0, 0] = -999.0
    assert outputs[0][0, 0] != -999.0


def test_batch_of_two_returns_two_entries() -> None:
    """A batch of 2 images produces 2 postprocessed entries."""
    batch = np.concatenate([_LOGITS, _LOGITS * 2.0], axis=0)
    result = postprocess_classifications([batch], softmax=True)

    assert len(result) == 2
    for entry in result:
        assert float(np.sum(entry[0])) == pytest.approx(1.0, abs=1e-5)
