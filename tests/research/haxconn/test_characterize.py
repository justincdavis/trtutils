# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/research/haxconn/ characterization, builds and runtime -- on real engines."""

from __future__ import annotations

import subprocess
import time

import numpy as np
import pytest

from trtutils import FLAGS, TRTEngine
from trtutils.builder import build_engine
from trtutils.core import Buffer
from trtutils.research.haxconn import (
    HaxConn,
    HaxConnConfig,
    HaxConnScheduler,
    calibrate,
    characterize,
    measure_memory,
    slice_hook,
)

_CONFIG = HaxConnConfig(contention="none", solver="exhaustive")


@pytest.fixture(scope="module")
def conv_model(conv_onnx):
    """Conv-ReLU-conv network that runs fully on a DLA: two groups, cut after the ReLU."""
    return conv_onnx(64, 56, 56)


@pytest.fixture(scope="module")
def conv_traits(conv_model):
    """FP16 traits of the conv network."""
    return characterize(conv_model, iterations=20, warmup_iterations=5)


def _sudo_tegrastats() -> bool:
    result = subprocess.run(["sudo", "-n", "true"], check=False, capture_output=True)  # noqa: S607
    return result.returncode == 0


def test_slice_hook_round_trip(conv_model, tmp_path) -> None:
    """Two sliced group engines chained compute what the whole network computes."""
    paths = {}
    for name, (first, last) in {"full": (0, 2), "head": (0, 1), "tail": (2, 2)}.items():
        paths[name] = tmp_path / f"{name}.engine"
        hooks = None if name == "full" else [slice_hook(first, last)]
        build_engine(conv_model, paths[name], hooks=hooks, optimization_level=1)
    full, head, tail = (TRTEngine(paths[n], warmup=False) for n in ("full", "head", "tail"))
    assert sorted(tail.input_names) == ["in_0", "r1__in"]
    assert head.output_names == ["r1"]
    data = np.random.default_rng(0).random((1, 64, 56, 56), dtype=np.float32)
    middle = head.execute({"in_0": Buffer.wrap(data)})[0]
    result = tail.execute({"in_0": Buffer.wrap(data), "r1__in": Buffer.wrap(middle)})[0]
    np.testing.assert_allclose(result, full.execute([Buffer.wrap(data)])[0], rtol=1e-3, atol=1e-4)
    with pytest.raises(ValueError, match="outside the network"):
        build_engine(conv_model, tmp_path / "bad.engine", hooks=[slice_hook(2, 5)])


def test_characterize_conv_network(conv_traits) -> None:
    """Characterization splits after the fused conv+ReLU and times every group."""
    assert [(g.first, g.last) for g in conv_traits.groups] == [(0, 1), (2, 2)]
    assert conv_traits.precision == "fp16"
    for group in conv_traits.groups:
        assert group.time("gpu") > 0.0
        assert group.dla is FLAGS.HAS_DLA
        if group.dla:
            assert group.time("dla") > 0.0
            assert group.transition_in_ms["dla"] >= conv_traits.dla_launch_ms


@pytest.mark.dla
def test_runtime_static_and_dynamic(conv_model, conv_traits, tmp_path) -> None:
    """The runtime runs the solved schedule, and dynamic mode reaches the same schedule."""
    models = {"a": conv_model, "b": conv_model}
    traits = {"a": conv_traits, "b": conv_traits}
    best = HaxConnScheduler(traits, config=_CONFIG).solve()
    runtime = HaxConn(models, traits, config=_CONFIG, engine_dir=tmp_path)
    assert runtime.schedule == best
    outputs = runtime.mock_execute()
    assert {name: out[0].shape for name, out in outputs.items()} == {
        "a": (1, 64, 56, 56),
        "b": (1, 64, 56, 56),
    }
    assert runtime.last is not None
    assert all(ms > 0.0 for ms in runtime.last.values())
    runtime.close()
    with pytest.raises(RuntimeError, match="closed"):
        runtime.mock_execute()

    dynamic = HaxConn(models, traits, config=_CONFIG, engine_dir=tmp_path, dynamic=True)
    deadline = time.monotonic() + 120.0
    while not dynamic.schedule.optimal and time.monotonic() < deadline:
        dynamic.mock_execute()
    assert dynamic.schedule == best
    dynamic.close()


@pytest.mark.jetson
def test_memory_and_calibration(conv_model, tmp_path) -> None:
    """Tegrastats reports EMC utilisation, and a small calibration fills both matrices."""
    if not _sudo_tegrastats():
        pytest.skip("memory throughput needs root or passwordless sudo for tegrastats")
    path = tmp_path / "conv.engine"
    build_engine(conv_model, path, optimization_level=1)
    engine = TRTEngine(path, cuda_graph=False, warmup=False)
    data = engine.get_random_input()
    samples = measure_memory(lambda: engine.execute(data), seconds=0.8)
    assert len(samples) > 0
    assert all(0.0 <= s <= 100.0 for s in samples)
    if not FLAGS.HAS_DLA:
        return
    calibration = calibrate(sizes=(32, 128), kernels=(1,), depth=2, seconds=1.0)
    for acc in ("gpu", "dla"):
        c = calibration[acc]
        assert len(c.demand) == len(c.external) == 2
        assert np.array(c.relative_speed).shape == (2, 2)
        assert all(0.0 < v <= 100.0 for row in c.relative_speed for v in row)
