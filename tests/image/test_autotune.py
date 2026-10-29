# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/_autotune.py -- timing runtime options of image models."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from trtutils.image import AutotuneTrial, Classifier, OBBDetector, Segmenter, autotune
from trtutils.image._autotune import _copy, _equal

_FAST = {"iterations": 1, "warmup": 1}


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        pytest.param(1.0, 1.0, True, id="float-equal"),
        pytest.param(1.0, 1.0 + 1e-9, False, id="float-differs"),
        pytest.param((1, 2.0), (1, 2.0), True, id="tuple-equal"),
        pytest.param([1, 2], [1, 2, 3], False, id="length-differs"),
        pytest.param(np.arange(4), np.arange(4), True, id="array-equal"),
        pytest.param(np.arange(4), np.arange(4) + 1, False, id="array-differs"),
        pytest.param(np.zeros(4), np.zeros(5), False, id="array-shape-differs"),
        pytest.param(
            [((1, 2, 3, 4), 0.5, np.ones(3))] * 2,
            [((1, 2, 3, 4), 0.5, np.ones(3))] * 2,
            True,
            id="nested-equal",
        ),
        pytest.param([(1, 0.5, np.ones(3))], [(1, 0.5, np.zeros(3))], False, id="nested-differs"),
    ],
)
def test_equal(a, b, expected: bool) -> None:
    """Equality is exact over nested lists, tuples, arrays and floats."""
    assert _equal(a, b) is expected


@pytest.mark.cpu
def test_copy_detaches_arrays() -> None:
    """Copied results do not alias the arrays they came from."""
    arr = np.zeros(3)
    out = _copy([(1, arr)])
    arr[0] = 5
    assert out[0][1][0] == 0
    assert isinstance(out, list)
    assert isinstance(out[0], tuple)


@pytest.mark.cpu
def test_trial_kwargs() -> None:
    """Kwargs holds the three constructor options."""
    trial = AutotuneTrial(
        preprocessor="cuda",
        postprocessor="cuda",
        cuda_graph=True,
        median_ms=1.0,
        p90_ms=1.0,
        same_as_default=True,
        error=None,
    )
    assert trial.kwargs == {
        "preprocessor": "cuda",
        "postprocessor": "cuda",
        "cuda_graph": True,
    }


@pytest.mark.parametrize(
    ("model_cls", "task", "posts"),
    [
        pytest.param(Classifier, "cls", ("cpu",), id="cls"),
        pytest.param(Segmenter, "seg", ("cpu", "cuda"), id="seg"),
        pytest.param(OBBDetector, "obb", ("cpu", "cuda"), id="obb"),
    ],
)
def test_default_grid_covers_all_options(task_engine, images, model_cls, task, posts) -> None:
    """With no options given every valid combination gets a trial, each once."""
    trials = autotune(model_cls, task_engine(task), [images["horse"].array], **_FAST)
    combos = {(t.preprocessor, t.postprocessor, t.cuda_graph) for t in trials}
    expected = set(itertools.product(("cpu", "cuda", "trt", "hybrid"), posts, (True, False)))
    assert len(trials) == len(expected)
    assert combos == expected


@pytest.mark.parametrize(
    ("model_cls", "task", "default_post"),
    [
        pytest.param(Classifier, "cls", "cpu", id="cls"),
        pytest.param(Segmenter, "seg", "cuda", id="seg"),
    ],
)
def test_sorted_and_default_matches_itself(
    task_engine, images, model_cls, task, default_post
) -> None:
    """Trials are fastest first, failures last, and the default configuration is the same."""
    engine = task_engine(task)
    trials = autotune(
        model_cls, engine, [images["horse"].array], postprocessors=["cpu", "cuda"], **_FAST
    )
    ok = [t for t in trials if t.error is None]
    failed = [t for t in trials if t.error is not None]
    assert trials == ok + failed
    assert [t.median_ms for t in ok] == sorted(t.median_ms for t in ok)
    for t in ok:
        assert 0 < t.median_ms <= t.p90_ms
        assert t.same_as_default is not None
    for t in failed:
        assert t.same_as_default is None
        assert t.median_ms == float("inf")
    default = next(
        t for t in ok if t.kwargs["preprocessor"] == "cuda" and t.postprocessor == default_post
    )
    assert default.same_as_default is True


def test_invalid_option_is_a_failed_trial(task_engine, images) -> None:
    """A postprocessor the class does not support records an error instead of raising."""
    trials = autotune(
        Classifier,
        task_engine("cls"),
        [images["horse"].array],
        preprocessors=["cuda"],
        postprocessors=["cuda"],
        cuda_graphs=[False],
        **_FAST,
    )
    assert len(trials) == 1
    assert trials[0].error is not None
    assert "postprocessor" in trials[0].error
    assert trials[0].same_as_default is None


@pytest.mark.parametrize("graph", [True, False])
def test_explicit_options_limit_the_grid(task_engine, images, graph: bool) -> None:
    """Explicit options are the only ones tried, and several images are timed."""
    img = images["horse"].array
    trials = autotune(
        Classifier,
        task_engine("cls"),
        [img, img[:200, :300]],
        preprocessors=["cpu", "cuda"],
        postprocessors=["cpu"],
        cuda_graphs=[graph],
        model_kwargs={"softmax": True},
        **_FAST,
    )
    assert {(t.preprocessor, t.cuda_graph) for t in trials} == {("cpu", graph), ("cuda", graph)}
    assert all(t.error is None for t in trials)


def test_kwargs_round_trip_into_constructor(task_engine, images) -> None:
    """The best trial's kwargs build a working model."""
    img = images["horse"].array
    engine = task_engine("seg")
    best = autotune(Segmenter, engine, [img], **_FAST)[0]
    assert best.error is None
    model = Segmenter(engine, warmup=False, **best.kwargs)
    assert len(model.end2end([img])) == 1
