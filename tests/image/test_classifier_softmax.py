# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/postprocessors/_classifier.py -- the softmax flag."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils.image.postprocessors import postprocess_classifications

pytestmark = pytest.mark.cpu

_LOGITS = np.array([[1.0, 2.0, 3.0, 0.5]], dtype=np.float32)


@pytest.mark.parametrize(
    "softmax", [pytest.param(True, id="softmax"), pytest.param(False, id="passthrough")]
)
@pytest.mark.parametrize("batch", [pytest.param(1, id="batch-1"), pytest.param(2, id="batch-2")])
def test_postprocess_classifications_softmax(softmax: bool, batch: int) -> None:
    """Each image gets a softmax distribution, or its raw values with softmax=False, as a copy."""
    logits = np.concatenate([_LOGITS * (i + 1) for i in range(batch)])
    outputs = [logits.copy()]
    result = postprocess_classifications(outputs, softmax=softmax)
    after = outputs[0].copy()

    assert len(result) == batch
    for row, entry in zip(logits, result):
        probs = entry[0].ravel()
        expected = np.exp(row) / np.exp(row).sum() if softmax else row
        np.testing.assert_allclose(probs, expected, rtol=1e-5)
        entry[0][...] = -999.0
    np.testing.assert_array_equal(outputs[0], after)
