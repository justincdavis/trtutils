# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/research/haxconn/_simulate.py -- the concurrent DNN timeline."""

from __future__ import annotations

import pytest

from tests.research.haxconn.conftest import XAVIER, make_traits
from trtutils.research.haxconn import HaxConnConfig, simulate

pytestmark = pytest.mark.cpu

_NONE = HaxConnConfig(contention="none")


def test_transitions_and_queueing() -> None:
    """Transitions delay a DNN, and DNNs on one accelerator queue in arrival order."""
    chain = make_traits([1.0, 2.0], [3.0, 1.0], to_dla=[0.5, 0.2], to_gpu=[0.1, 0.3])
    # 0.1 in + 1.0 on the GPU + 0.2 into the DLA + 1.0 on the DLA
    assert simulate([chain], [["gpu", "dla"]], config=_NONE) == pytest.approx([2.3])
    assert simulate([chain], [["dla", "dla"]], config=_NONE) == pytest.approx([4.5])
    first, second = make_traits([1.0], [None]), make_traits([2.0], [None])
    assert simulate([first, second], [["gpu"], ["gpu"]], config=_NONE) == pytest.approx([1.0, 3.0])


def test_segments_hold_their_accelerator() -> None:
    """A run of groups on one accelerator keeps it, so a waiting DNN starts after the run."""
    long = make_traits([1.0, 1.0], [None, None])
    short = make_traits([0.5], [None])
    assert simulate([long, short], [["gpu", "gpu"], ["gpu"]], config=_NONE) == pytest.approx(
        [2.0, 2.5]
    )


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        pytest.param(_NONE, [2.0, 2.0], id="none"),
        pytest.param(HaxConnConfig(contention="overlap"), "overlap", id="overlap"),
        pytest.param(HaxConnConfig(contention="static"), "static", id="static"),
    ],
)
def test_contention_modes(config, expected) -> None:
    """Overlap slows groups only while both accelerators run; static always slows them."""
    on_gpu = make_traits([2.0], [2.0], gpu_memory=[60.0], dla_memory=[60.0])
    on_dla = make_traits([2.0], [2.0], gpu_memory=[40.0], dla_memory=[40.0])
    gpu, dla = XAVIER["gpu"].slowdown(60.0, 40.0), XAVIER["dla"].slowdown(40.0, 60.0)
    if expected == "overlap":
        # the GPU group ends first; the DLA group finishes its remaining work alone
        done = 2.0 * gpu
        expected = [done, done + 2.0 - done / dla]
    elif expected == "static":
        expected = [
            2.0 * XAVIER["gpu"].slowdown(60.0, 100.0),
            2.0 * XAVIER["dla"].slowdown(40.0, 100.0),
        ]
    result = simulate([on_gpu, on_dla], [["gpu"], ["dla"]], XAVIER, config)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    ("assignments", "contention", "config", "match"),
    [
        pytest.param([["gpu"]], None, _NONE, "expected 2 accelerators", id="length"),
        pytest.param([["gpu", "dla"]], None, _NONE, "cannot run on 'dla'", id="gpu-only-group"),
        pytest.param([["gpu", "gpu"]], None, HaxConnConfig(), "needs a contention", id="contention"),
    ],
)
def test_simulate_rejects_invalid_schedules(assignments, contention, config, match) -> None:
    """Wrong lengths, impossible accelerators and missing contention models raise."""
    traits = make_traits([1.0, 1.0], [1.0, None])
    with pytest.raises(ValueError, match=match):
        simulate([traits], assignments, contention, config)
