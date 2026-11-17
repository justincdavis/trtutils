# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for build_engine() -- all parameter combinations and branch coverage."""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.builder.conftest import ListBatcher
from trtutils import Buffer, TRTEngine
from trtutils._flags import FLAGS as REAL_FLAGS
from trtutils._log import LOG
from trtutils.builder import _build
from trtutils.builder._batcher import SyntheticBatcher
from trtutils.builder._build import build_engine
from trtutils.builder.onnx._shapes import get_onnx_input, get_onnx_output
from trtutils.compat._libs import trt
from trtutils.core import cache as caching_tools
from trtutils.inspect import inspect_engine

_BLACKWELL_WEAK_FAIL_REASON = (
    "Weakly-typed INT8/FP8 mixes require strongly_typed=True on Blackwell "
    "(SM 10.0+); see test_build_strongly_typed_from_qdq for the replacement path."
)
# tensorrt 8.6 needs real calibration scales for int8 and q/dq layers for fp8
_needs_trt10_int8 = pytest.mark.skipif(
    REAL_FLAGS.TRT_VERSION < (10, 0), reason="int8 without calibration scales needs TensorRT 10"
)
_needs_weak_typing = pytest.mark.skipif(
    not REAL_FLAGS.WEAK_TYPING, reason="weakly-typed networks were removed in TensorRT 11"
)
_needs_strong_typing = pytest.mark.skipif(
    not REAL_FLAGS.STRONGLY_TYPED_SUPPORTED, reason="strongly-typed networks need TensorRT 10"
)
_needs_trt10_fp8 = pytest.mark.skipif(
    REAL_FLAGS.TRT_VERSION < (10, 0), reason="weakly-typed fp8 needs TensorRT 10"
)
# (min, opt, max) shapes for dynamic_batch_onnx_path, calibrated at opt
_DYNAMIC_BATCH_PROFILE = ((1, 3, 16, 16), (2, 3, 16, 16), (4, 3, 16, 16))


@pytest.mark.parametrize(
    ("fp16", "int8", "fp8"),
    [
        pytest.param(False, False, False, id="default"),
        pytest.param(True, False, False, id="fp16"),
        pytest.param(
            False,
            True,
            False,
            id="int8",
            marks=[
                pytest.mark.skipif(REAL_FLAGS.IS_BLACKWELL, reason=_BLACKWELL_WEAK_FAIL_REASON),
                _needs_trt10_int8,
            ],
        ),
        pytest.param(False, False, True, id="fp8", marks=_needs_trt10_fp8),
        pytest.param(
            True,
            True,
            False,
            id="fp16-int8",
            marks=[
                pytest.mark.skipif(REAL_FLAGS.IS_BLACKWELL, reason=_BLACKWELL_WEAK_FAIL_REASON),
                _needs_trt10_int8,
            ],
        ),
        pytest.param(True, False, True, id="fp16-fp8", marks=_needs_trt10_fp8),
        pytest.param(
            False,
            True,
            True,
            id="int8-fp8",
            marks=[
                pytest.mark.skipif(REAL_FLAGS.IS_BLACKWELL, reason=_BLACKWELL_WEAK_FAIL_REASON),
                _needs_trt10_fp8,
            ],
        ),
        pytest.param(
            True,
            True,
            True,
            id="all",
            marks=[
                pytest.mark.skipif(REAL_FLAGS.IS_BLACKWELL, reason=_BLACKWELL_WEAK_FAIL_REASON),
                _needs_trt10_fp8,
            ],
        ),
    ],
)
@_needs_weak_typing
def test_build_precision(
    onnx_path,
    output_engine_path,
    fp16: bool,
    int8: bool,
    fp8: bool,
) -> None:
    """Build succeeds with all precision flag combinations."""
    build_engine(
        onnx_path,
        output_engine_path,
        fp16=fp16,
        int8=int8,
        fp8=fp8,
        optimization_level=1,
    )
    assert output_engine_path.exists()
    assert output_engine_path.stat().st_size > 0


@pytest.mark.skipif(
    REAL_FLAGS.IS_BLACKWELL,
    reason="Runtime INT8/FP8 calibration is a weakly-typed concept; see test_build_strongly_typed_from_qdq.",
)
@pytest.mark.parametrize(
    ("precision", "calibration"),
    [
        pytest.param("int8", "none", id="int8-no-calibration", marks=_needs_trt10_int8),
        pytest.param("int8", "batcher", id="int8-batcher"),
        pytest.param("int8", "cache", id="int8-cache", marks=_needs_trt10_int8),
        pytest.param("fp8", "none", id="fp8-no-calibration", marks=_needs_trt10_fp8),
        pytest.param("fp8", "batcher", id="fp8-batcher", marks=_needs_trt10_fp8),
        pytest.param("fp8", "cache", id="fp8-cache", marks=_needs_trt10_fp8),
    ],
)
@_needs_weak_typing
def test_build_precision_calibration(
    onnx_path,
    output_engine_path,
    calibration_cache_path,
    synthetic_batcher,
    precision: str,
    calibration: str,
) -> None:
    """Build succeeds with int8/fp8 using no calibration, batcher, or calibration cache."""
    kwargs: dict = {precision: True, "optimization_level": 1}
    if calibration == "batcher":
        kwargs["data_batcher"] = synthetic_batcher
    elif calibration == "cache":
        kwargs["calibration_cache"] = calibration_cache_path
    build_engine(onnx_path, output_engine_path, **kwargs)
    assert output_engine_path.exists()
    assert output_engine_path.stat().st_size > 0


@pytest.mark.skipif(
    REAL_FLAGS.IS_BLACKWELL,
    reason="Runtime INT8/FP8 calibration is a weakly-typed concept; see test_build_strongly_typed_from_qdq.",
)
@pytest.mark.parametrize(
    "ranged",
    [
        pytest.param("all", id="every-tensor-ranged"),
        pytest.param("inputs", id="input-ranged-only"),
    ],
)
@_needs_weak_typing
def test_build_int8_dynamic_ranges_without_calibration(
    onnx_path,
    output_engine_path,
    tmp_path,
    monkeypatch,
    ranged: str,
) -> None:
    """int8 without calibration data attaches no calibrator and uses the set dynamic ranges."""
    counts: dict[str, int] = {}

    def set_ranges(network: trt.INetworkDefinition) -> trt.INetworkDefinition:
        inputs = [network.get_input(idx) for idx in range(network.num_inputs)]
        outputs = [
            network.get_layer(idx).get_output(out)
            for idx in range(network.num_layers)
            for out in range(network.get_layer(idx).num_outputs)
        ]
        tensors = inputs + outputs if ranged == "all" else inputs
        for tensor in tensors:
            tensor.set_dynamic_range(-2.0, 2.0)
        counts["total"] = len(inputs) + sum(not t.is_network_output for t in outputs)
        counts["missing"] = counts["total"] - len(tensors)
        return network

    # a calibrator without a cache path writes a temporary file
    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_dir))
    with patch.object(_build, "EngineCalibrator", wraps=_build.EngineCalibrator) as calibrator:
        with patch.object(LOG, "warning", wraps=LOG.warning) as warning:
            build_engine(
                onnx_path,
                output_engine_path,
                int8=True,
                hooks=[set_ranges],
                optimization_level=1,
            )

    assert output_engine_path.stat().st_size > 0
    calibrator.assert_not_called()
    assert list(temp_dir.iterdir()) == []
    messages = [str(call.args[0]) for call in warning.call_args_list if call.args]
    assert not [m for m in messages if "Neither calibration cache" in m or "temporary file" in m]
    missing = [m for m in messages if m.startswith("INT8 without calibration data")]
    if ranged == "all":
        assert missing == []
    else:
        assert counts["missing"] > 0
        assert len(missing) == 1
        assert (
            f"{counts['missing']} of {counts['total']} tensors have no dynamic range" in missing[0]
        )


@pytest.mark.skipif(
    REAL_FLAGS.IS_BLACKWELL,
    reason="Runtime INT8/FP8 calibration is a weakly-typed concept; see test_build_strongly_typed_from_qdq.",
)
@_needs_weak_typing
def test_int8_calibration_casts_batches_to_input_dtype(
    onnx_path,
    output_engine_path,
    calibration_cache_path,
) -> None:
    """float32 batches for a float16 input are cast, not read as float16 bytes."""

    def fp16_input(network: trt.INetworkDefinition) -> trt.INetworkDefinition:
        network.get_input(0).dtype = trt.float16
        return network

    batcher = SyntheticBatcher(
        shape=(160, 160, 3),
        dtype=np.dtype(np.float32),
        batch_size=1,
        num_batches=2,
        order="NCHW",
    )
    build_engine(
        onnx_path,
        output_engine_path,
        int8=True,
        data_batcher=batcher,
        calibration_cache=calibration_cache_path,
        hooks=[fp16_input],
        optimization_level=1,
    )
    scale = _calibration_scales(calibration_cache_path)["input"]
    # synthetic data lies in [0, 1], so the input scale is about 1/127. Read as
    # float16 bytes, the float32 data held inf/nan and gave an infinite scale.
    assert 0.0 < scale < 0.02


@pytest.mark.skipif(
    REAL_FLAGS.IS_BLACKWELL,
    reason="Runtime INT8/FP8 calibration is a weakly-typed concept; see test_build_strongly_typed_from_qdq.",
)
@pytest.mark.parametrize(
    "batch_dtype",
    [
        pytest.param(np.float32, id="float32-batches"),
        pytest.param(np.int8, id="int8-batches"),
    ],
)
@_needs_weak_typing
def test_int8_calibration_of_int8_input(
    onnx_path,
    output_engine_path,
    calibration_cache_path,
    batch_dtype,
) -> None:
    """Batches for an int8 input are copied as float32, the dtype TensorRT binds it as."""

    def int8_input(network: trt.INetworkDefinition) -> trt.INetworkDefinition:
        tensor = network.get_input(0)
        tensor.dtype = trt.int8
        tensor.set_dynamic_range(-127.0, 127.0)
        return network

    batcher = SyntheticBatcher(
        shape=(160, 160, 3),
        dtype=np.dtype(batch_dtype),
        batch_size=1,
        num_batches=2,
        data_range=(-100.0, 100.0),
        order="NCHW",
    )
    build_engine(
        onnx_path,
        output_engine_path,
        int8=True,
        data_batcher=batcher,
        calibration_cache=calibration_cache_path,
        hooks=[int8_input],
        optimization_level=1,
    )
    scale = _calibration_scales(calibration_cache_path)["output"]
    # the network computes x * 0.5 + 1, so data in [-100, 100] gives |output| <= 51
    # and an output scale of about 51/127 = 0.40 (entropy calibration can land up to
    # ~25% above the amax). Copied as int8, the batch was read as float32 bytes (4x
    # past its end) and gave scales around 1e36.
    assert 0.3 < scale < 0.8


@pytest.mark.skipif(
    REAL_FLAGS.IS_BLACKWELL,
    reason="Runtime INT8/FP8 calibration is a weakly-typed concept; see test_build_strongly_typed_from_qdq.",
)
@pytest.mark.parametrize(
    "kind",
    [
        pytest.param("list", id="list-batches"),
        pytest.param("dict", id="dict-batches"),
    ],
)
@_needs_weak_typing
def test_int8_calibration_of_two_inputs(
    two_input_onnx_path,
    output_engine_path,
    calibration_cache_path,
    kind,
) -> None:
    """A two-input network calibrates from list or dict batches, each input on its own data."""
    rng = np.random.default_rng(0)
    batches: list = []
    for _ in range(4):
        a = rng.uniform(0.0, 1.0, (1, 3, 16, 16)).astype(np.float32)
        # float64 batches for b are cast to float32
        b = rng.uniform(-4.0, 4.0, (1, 3, 16, 16))
        batches.append([a, b] if kind == "list" else {"b": b, "a": a})
    build_engine(
        two_input_onnx_path,
        output_engine_path,
        int8=True,
        data_batcher=ListBatcher(batches),
        calibration_cache=calibration_cache_path,
        optimization_level=1,
    )
    assert output_engine_path.exists()
    scales = _calibration_scales(calibration_cache_path)
    # a in [0, 1] and b in [-4, 4] give input scales of about 1/127 and 4/127, which
    # swapped inputs or one buffer for both would not
    assert 0.006 < scales["a"] < 0.012
    assert 0.025 < scales["b"] < 0.05
    # y sums the three channels of a + b, so |y| <= 15 and its scale is at most 15/127
    assert 0.0 < scales["y"] < 0.15


@pytest.mark.skipif(
    REAL_FLAGS.IS_BLACKWELL,
    reason="Runtime INT8/FP8 calibration is a weakly-typed concept; see test_build_strongly_typed_from_qdq.",
)
@_needs_weak_typing
def test_int8_calibration_of_dynamic_input(
    dynamic_batch_onnx_path,
    output_engine_path,
    calibration_cache_path,
) -> None:
    """A dynamic input calibrates on whole batches of the opt shape in its profile."""
    batch = np.empty((2, 3, 16, 16), dtype=np.float32)
    batch[0] = np.linspace(0.0, 1.0, batch[0].size).reshape(batch[0].shape)
    batch[1] = 100.0 * batch[0]
    build_engine(
        dynamic_batch_onnx_path,
        output_engine_path,
        int8=True,
        data_batcher=ListBatcher([batch, batch]),
        calibration_cache=calibration_cache_path,
        shapes=[("x", _DYNAMIC_BATCH_PROFILE)],
        optimization_level=1,
    )
    # the second sample spans [0, 100], so reading it gives a scale of about 100/127,
    # not the 1/127 of the first sample alone
    assert 0.6 < _calibration_scales(calibration_cache_path)["x"] < 1.0


@pytest.mark.skipif(
    REAL_FLAGS.IS_BLACKWELL,
    reason="Runtime INT8/FP8 calibration is a weakly-typed concept; see test_build_strongly_typed_from_qdq.",
)
@pytest.mark.parametrize(
    ("onnx", "batch", "shapes", "match"),
    [
        pytest.param(
            "two_input_onnx_path",
            [np.zeros((1, 3, 16, 16), dtype=np.float32)],
            None,
            r"network has 2 inputs \['a', 'b'\]",
            id="one-array-for-two-inputs",
        ),
        # 1/400 of the (1, 3, 160, 160) input, which tensorrt would read past
        pytest.param(
            "onnx_path",
            np.zeros((1, 3, 8, 8), dtype=np.float32),
            None,
            r"TensorRT reads 76800 values for its calibration shape \(1, 3, 160, 160\)",
            id="array-smaller-than-input",
        ),
        pytest.param(
            "dynamic_batch_onnx_path",
            np.zeros((1, 3, 16, 16), dtype=np.float32),
            [("x", _DYNAMIC_BATCH_PROFILE)],
            r"TensorRT reads 1536 values for its calibration shape \(2, 3, 16, 16\)",
            id="array-smaller-than-opt-shape",
        ),
    ],
)
@_needs_weak_typing
def test_int8_calibration_batch_mismatch_raises(
    request,
    output_engine_path,
    calibration_cache_path,
    onnx: str,
    batch,
    shapes,
    match: str,
) -> None:
    """A batch that does not fit the network inputs fails the build with its ValueError."""
    with pytest.raises(ValueError, match=match):
        build_engine(
            request.getfixturevalue(onnx),
            output_engine_path,
            int8=True,
            data_batcher=ListBatcher([batch, batch]),
            calibration_cache=calibration_cache_path,
            shapes=shapes,
            optimization_level=1,
        )
    assert not output_engine_path.exists()


def _calibration_scales(cache: Path) -> dict[str, float]:
    """Per-tensor scales of a TensorRT calibration cache (big-endian float32 hex)."""
    lines = cache.read_text().splitlines()[1:]
    return {
        name: struct.unpack(">f", bytes.fromhex(value))[0]
        for name, value in (line.split(": ") for line in lines)
    }


def test_build_cache(onnx_path, output_engine_path) -> None:
    """Build with cache=True stores engine in cache; second call hits cache without invoking TRT."""
    # first build: engine is built and stored in the cache
    build_engine(onnx_path, output_engine_path, cache=True, optimization_level=1)
    assert output_engine_path.stat().st_size > 0

    cached_exists, cached_path = caching_tools.query(output_engine_path.stem)
    assert cached_exists
    assert cached_path.stat().st_size > 0

    # second build: cache hit should copy from cache without parsing the ONNX at all
    output_engine_path.unlink()
    with patch("trtutils.builder._build.read_onnx") as mock_read:
        build_engine(onnx_path, output_engine_path, cache=True, optimization_level=1)
        mock_read.assert_not_called()
    assert output_engine_path.exists()


def test_build_writes_engine_atomically(onnx_path, output_engine_path) -> None:
    """Final path only appears once complete, via replace from a same-dir temp file."""
    seen = []
    real_replace = Path.replace

    def spy_replace(self, target):
        seen.append((self.parent, self.stat().st_size, Path(target).exists()))
        return real_replace(self, target)

    with patch.object(Path, "replace", spy_replace):
        build_engine(onnx_path, output_engine_path, optimization_level=1)

    assert seen == [(output_engine_path.parent, output_engine_path.stat().st_size, False)]
    assert [p.name for p in output_engine_path.parent.iterdir()] == [output_engine_path.name]


@pytest.mark.parametrize(
    ("timing_cache_val", "valid"),
    [
        pytest.param(True, True, id="bool-true"),
        pytest.param("global", True, id="global-str"),
        pytest.param("local", True, id="local-file"),
        pytest.param(42, False, id="invalid-int"),
    ],
)
def test_timing_cache_modes(
    onnx_path, output_engine_path, timing_cache_path, timing_cache_val, valid: bool
) -> None:
    """Valid timing_cache values succeed; invalid types raise ValueError."""
    tc = timing_cache_path if timing_cache_val == "local" else timing_cache_val
    if valid:
        build_engine(
            onnx_path,
            output_engine_path,
            timing_cache=tc,
            optimization_level=1,
        )
        assert output_engine_path.exists()
        if timing_cache_val == "local":
            assert timing_cache_path.exists()
            assert timing_cache_path.stat().st_size > 0
    else:
        with pytest.raises(ValueError, match="Invalid timing_cache value"):
            build_engine(
                onnx_path,
                output_engine_path,
                timing_cache=tc,  # type: ignore[arg-type]
                optimization_level=1,
            )


@pytest.mark.parametrize(
    ("device", "valid"),
    [
        pytest.param("gpu", True, id="gpu-str-lower"),
        pytest.param("GPU", True, id="gpu-str-upper"),
        pytest.param(trt.DeviceType.GPU, True, id="gpu-enum"),
        pytest.param(
            "dla",
            True,
            id="dla-str-lower",
            marks=pytest.mark.skipif(not REAL_FLAGS.IS_JETSON, reason="DLA requires Jetson"),
        ),
        pytest.param(
            "DLA",
            True,
            id="dla-str-upper",
            marks=pytest.mark.skipif(not REAL_FLAGS.IS_JETSON, reason="DLA requires Jetson"),
        ),
        pytest.param(
            trt.DeviceType.DLA,
            True,
            id="dla-enum",
            marks=pytest.mark.skipif(not REAL_FLAGS.IS_JETSON, reason="DLA requires Jetson"),
        ),
        pytest.param("tpu", False, id="invalid-str"),
        pytest.param(999, False, id="invalid-int"),
    ],
)
def test_device_variants(onnx_path, output_engine_path, device, valid: bool) -> None:
    """Valid device strings and enums succeed; invalid values raise."""
    if valid:
        is_dla = (isinstance(device, str) and device.lower() == "dla") or (
            not isinstance(device, str) and device == trt.DeviceType.DLA
        )
        build_engine(
            onnx_path,
            output_engine_path,
            default_device=device,
            gpu_fallback=is_dla,
            # dla runs only fp16 or int8 layers
            fp16=is_dla,
            optimization_level=1,
        )
        assert output_engine_path.exists()
    else:
        with pytest.raises((ValueError, AttributeError)):
            build_engine(
                onnx_path,
                output_engine_path,
                default_device=device,  # type: ignore[arg-type]
                optimization_level=1,
            )


@pytest.mark.parametrize(
    ("level", "valid"),
    [
        pytest.param(0, True, id="min"),
        pytest.param(5, True, id="max"),
        pytest.param(-1, False, id="negative"),
        pytest.param(6, False, id="too-high"),
    ],
)
def test_optimization_level(onnx_path, output_engine_path, level: int, valid: bool) -> None:
    """Boundary optimization levels succeed; out-of-range levels raise ValueError."""
    if valid:
        build_engine(onnx_path, output_engine_path, optimization_level=level)
        assert output_engine_path.exists()
    else:
        with pytest.raises(ValueError, match="Builder optimization level must be between 0 and 5"):
            build_engine(onnx_path, output_engine_path, optimization_level=level)


def test_manual_shapes(onnx_path, output_engine_path) -> None:
    """Passing shapes sets min/opt/max profile for the model's actual input."""
    input_name, input_shape = get_onnx_input(onnx_path)
    build_engine(
        onnx_path,
        output_engine_path,
        shapes=[(input_name, input_shape)],
        optimization_level=1,
    )
    assert output_engine_path.exists()


def test_single_hook(onnx_path, output_engine_path) -> None:
    """A single hook function is called during build."""
    hook_called = []

    def identity_hook(network):
        hook_called.append(True)
        return network

    build_engine(
        onnx_path,
        output_engine_path,
        hooks=[identity_hook],
        optimization_level=1,
    )
    assert len(hook_called) == 1
    assert output_engine_path.exists()


def test_multiple_hooks(onnx_path, output_engine_path) -> None:
    """Multiple hooks are called in order."""
    call_order = []

    def hook_a(network):
        call_order.append("a")
        return network

    def hook_b(network):
        call_order.append("b")
        return network

    build_engine(
        onnx_path,
        output_engine_path,
        hooks=[hook_a, hook_b],
        optimization_level=1,
    )
    assert call_order == ["a", "b"]


def test_build_from_network_definition(output_engine_path) -> None:
    """A function that defines the network in code builds an engine computing that network."""
    weights = np.zeros((3, 3, 1, 1), dtype=np.float32)
    weights[[0, 1, 2], [2, 1, 0]] = 2.0
    bias = np.ones(3, dtype=np.float32)

    def define(network, config):
        assert isinstance(config, trt.IBuilderConfig)
        tensor = network.add_input("input", trt.float32, (1, 3, 8, 8))
        layer = network.add_convolution_nd(
            tensor, 3, (1, 1), trt.Weights(weights), trt.Weights(bias)
        )
        network.mark_output(layer.get_output(0))
        return weights, bias

    build_engine(define, output_engine_path, optimization_level=1)
    engine = TRTEngine(output_engine_path, warmup=False)
    data = np.random.default_rng(0).random((1, 3, 8, 8), dtype=np.float32)
    result = engine.execute([Buffer.wrap(data)])[0]
    np.testing.assert_allclose(result, 2.0 * data[:, ::-1] + 1.0, rtol=1e-5)


@pytest.mark.parametrize("valid", [True, False], ids=["valid", "invalid"])
def test_input_tensor_format(onnx_path, output_engine_path, valid: bool) -> None:
    """Valid input tensor name sets format; invalid name raises ValueError."""
    name = get_onnx_input(onnx_path)[0] if valid else "nonexistent"
    if valid:
        build_engine(
            onnx_path,
            output_engine_path,
            input_tensor_formats=[(name, trt.DataType.FLOAT, trt.TensorFormat.LINEAR)],
            optimization_level=1,
        )
        assert output_engine_path.exists()
    else:
        with pytest.raises(ValueError, match="not found in network"):
            build_engine(
                onnx_path,
                output_engine_path,
                input_tensor_formats=[(name, trt.DataType.FLOAT, trt.TensorFormat.LINEAR)],
                optimization_level=1,
            )


@pytest.mark.parametrize("valid", [True, False], ids=["valid", "invalid"])
def test_output_tensor_format(onnx_path, output_engine_path, valid: bool) -> None:
    """Valid output tensor name sets format; invalid name raises ValueError."""
    name = get_onnx_output(onnx_path)[0] if valid else "nonexistent"
    if valid:
        build_engine(
            onnx_path,
            output_engine_path,
            output_tensor_formats=[(name, trt.DataType.FLOAT, trt.TensorFormat.LINEAR)],
            optimization_level=1,
        )
        assert output_engine_path.exists()
    else:
        with pytest.raises(ValueError, match="not found in network"):
            build_engine(
                onnx_path,
                output_engine_path,
                output_tensor_formats=[(name, trt.DataType.FLOAT, trt.TensorFormat.LINEAR)],
                optimization_level=1,
            )


@pytest.mark.regression
@pytest.mark.parametrize(
    "kind",
    [pytest.param("input", id="input"), pytest.param("output", id="output")],
)
@pytest.mark.parametrize(
    ("typing", "dtype", "expected"),
    [
        pytest.param(
            "weak", trt.DataType.HALF, trt.DataType.HALF, id="weak-sets", marks=_needs_weak_typing
        ),
        pytest.param("no-weak", trt.DataType.FLOAT, trt.DataType.FLOAT, id="no-weak-same"),
        pytest.param("no-weak", trt.DataType.HALF, None, id="no-weak-differs"),
        pytest.param(
            "strong",
            trt.DataType.HALF,
            None,
            id="strongly-typed-differs",
            marks=_needs_strong_typing,
        ),
    ],
)
def test_tensor_format_dtype(
    monkeypatch, onnx_path, output_engine_path, kind, typing, dtype, expected
) -> None:
    """Weak typing sets a tensor-format dtype; strong typing raises on one the graph lacks."""
    # tensorrt 11 has no weak typing, so this takes its branch
    if typing == "no-weak":
        monkeypatch.setattr(REAL_FLAGS, "WEAK_TYPING", False)
    name = (get_onnx_input if kind == "input" else get_onnx_output)(onnx_path)[0]
    kwargs = {
        f"{kind}_tensor_formats": [(name, dtype, trt.TensorFormat.LINEAR)],
        "strongly_typed": typing == "strong",
        "optimization_level": 1,
    }
    if expected is None:
        with pytest.raises(ValueError, match=f"{kind}_tensor_formats requests DataType.HALF"):
            build_engine(onnx_path, output_engine_path, **kwargs)
        assert not output_engine_path.exists()
        return
    build_engine(onnx_path, output_engine_path, **kwargs)
    _, _, inputs, outputs = inspect_engine(output_engine_path)
    tensors = inputs if kind == "input" else outputs
    assert [(n, d) for n, _, d, _ in tensors] == [(name, expected)]


@pytest.mark.regression
def test_tensor_format_needing_reformat(onnx_path, output_engine_path) -> None:
    """A tensor format the first layer cannot consume builds through a reformat."""
    name = get_onnx_input(onnx_path)[0]
    # the elementwise layers of simple.onnx have no channel-last kernel
    build_engine(
        onnx_path,
        output_engine_path,
        input_tensor_formats=[(name, trt.DataType.FLOAT, trt.TensorFormat.HWC)],
        optimization_level=1,
    )
    _, _, inputs, _ = inspect_engine(output_engine_path)
    assert [(n, fmt) for n, _, _, fmt in inputs] == [(name, trt.TensorFormat.HWC)]


@pytest.mark.parametrize(
    ("layer_precision", "layer_device", "extra_kwargs"),
    [
        pytest.param([(0, trt.DataType.HALF)], None, {"fp16": True}, id="precision-half"),
        pytest.param([(0, None)], None, {}, id="precision-none-skip"),
        pytest.param(None, [(0, trt.DeviceType.GPU)], {}, id="device-gpu"),
        pytest.param(None, [(0, None)], {}, id="device-none-skip"),
    ],
)
@_needs_weak_typing
def test_layer_settings(
    onnx_path, output_engine_path, layer_precision, layer_device, extra_kwargs
) -> None:
    """Per-layer precision and device assignments build successfully."""
    build_engine(
        onnx_path,
        output_engine_path,
        layer_precision=layer_precision,
        layer_device=layer_device,
        optimization_level=1,
        **extra_kwargs,
    )
    assert output_engine_path.exists()


@pytest.mark.parametrize(
    ("onnx_fixture", "exc"),
    [
        pytest.param("invalid_onnx_file", RuntimeError, id="invalid-content"),
        pytest.param(None, FileNotFoundError, id="nonexistent"),
    ],
)
def test_bad_onnx_raises(request, output_engine_path, tmp_path, onnx_fixture, exc) -> None:
    """Invalid or missing ONNX path raises the expected exception."""
    path = request.getfixturevalue(onnx_fixture) if onnx_fixture else tmp_path / "nonexistent.onnx"
    with pytest.raises(exc):
        build_engine(path, output_engine_path, optimization_level=1)


def test_build_failure_raises(onnx_path, output_engine_path) -> None:
    """Engine build returning None raises RuntimeError."""
    with patch("trtutils.builder._build.read_onnx") as mock_read:
        mock_builder = MagicMock()
        mock_read.return_value = (MagicMock(), mock_builder, MagicMock(), MagicMock())
        mock_builder.build_serialized_network.return_value = None
        mock_builder.build_engine.return_value = None
        mock_builder.create_optimization_profile.return_value = MagicMock()

        with pytest.raises(RuntimeError, match="Failed to build engine"):
            build_engine(onnx_path, output_engine_path, optimization_level=1)


def test_build_strongly_typed_from_qdq(quantized_onnx_path, output_engine_path) -> None:
    """Build from a pre-quantized Q/DQ ONNX with strongly_typed=True produces an engine."""
    build_engine(
        quantized_onnx_path,
        output_engine_path,
        strongly_typed=True,
        optimization_level=1,
    )
    assert output_engine_path.exists()
    assert output_engine_path.stat().st_size > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"fp16": True}, id="with-fp16"),
        pytest.param({"int8": True}, id="with-int8"),
        pytest.param({"fp8": True}, id="with-fp8"),
        pytest.param({"layer_precision": [(0, trt.DataType.HALF)]}, id="with-layer-precision"),
    ],
)
def test_build_strongly_typed_rejects_weak_args(
    quantized_onnx_path, output_engine_path, kwargs
) -> None:
    """strongly_typed=True combined with precision or per-layer overrides raises ValueError."""
    with pytest.raises(ValueError, match="strongly_typed"):
        build_engine(
            quantized_onnx_path,
            output_engine_path,
            strongly_typed=True,
            optimization_level=1,
            **kwargs,
        )


def test_build_strongly_typed_rejects_calibration_args(
    quantized_onnx_path,
    output_engine_path,
    calibration_cache_path,
    synthetic_batcher,
) -> None:
    """strongly_typed=True with calibration_cache or data_batcher raises ValueError."""
    with pytest.raises(ValueError, match="runtime calibration"):
        build_engine(
            quantized_onnx_path,
            output_engine_path,
            strongly_typed=True,
            calibration_cache=calibration_cache_path,
            optimization_level=1,
        )
    with pytest.raises(ValueError, match="runtime calibration"):
        build_engine(
            quantized_onnx_path,
            output_engine_path,
            strongly_typed=True,
            data_batcher=synthetic_batcher,
            optimization_level=1,
        )


@pytest.mark.skipif(
    not REAL_FLAGS.IS_BLACKWELL,
    reason="Blackwell-only helpful-error check",
)
@_needs_weak_typing
def test_build_blackwell_int8_fp8_rejects_weakly_typed(onnx_path, output_engine_path) -> None:
    """On Blackwell, int8+fp8 without strongly_typed=True raises with a docs pointer."""
    with pytest.raises(ValueError, match=r"strongly_typed=True.*strongly_typed\.rst"):
        build_engine(
            onnx_path,
            output_engine_path,
            int8=True,
            fp8=True,
            optimization_level=1,
        )
