# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for build_engine() -- all parameter combinations and branch coverage."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from trtutils._flags import FLAGS as REAL_FLAGS
from trtutils.builder._batcher import SyntheticBatcher
from trtutils.builder._build import build_engine
from trtutils.builder.onnx._shapes import get_onnx_input, get_onnx_output
from trtutils.compat._libs import trt
from trtutils.core import cache as caching_tools


@pytest.mark.parametrize(
    ("fp16", "int8", "fp8"),
    [
        pytest.param(False, False, False, id="default"),
        pytest.param(True, False, False, id="fp16"),
        pytest.param(False, True, False, id="int8"),
        pytest.param(False, False, True, id="fp8"),
        pytest.param(True, True, False, id="fp16-int8"),
        pytest.param(True, False, True, id="fp16-fp8"),
        pytest.param(False, True, True, id="int8-fp8"),
        pytest.param(True, True, True, id="all"),
    ],
)
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


@pytest.mark.parametrize(
    ("precision", "calibration"),
    [
        pytest.param("int8", "none", id="int8-no-calibration"),
        pytest.param("int8", "batcher", id="int8-batcher"),
        pytest.param("int8", "cache", id="int8-cache"),
        pytest.param("fp8", "none", id="fp8-no-calibration"),
        pytest.param("fp8", "batcher", id="fp8-batcher"),
        pytest.param("fp8", "cache", id="fp8-cache"),
    ],
)
def test_build_precision_calibration(
    onnx_path,
    output_engine_path,
    calibration_cache_path,
    precision: str,
    calibration: str,
) -> None:
    """Build succeeds with int8/fp8 using no calibration, batcher, or calibration cache."""
    kwargs: dict = {precision: True, "optimization_level": 1}
    if calibration == "batcher":
        kwargs["data_batcher"] = SyntheticBatcher(
            shape=(3, 8, 8),
            dtype=np.dtype(np.float32),
            batch_size=1,
            num_batches=2,
            order="NCHW",
        )
    elif calibration == "cache":
        kwargs["calibration_cache"] = calibration_cache_path
    build_engine(onnx_path, output_engine_path, **kwargs)
    assert output_engine_path.exists()
    assert output_engine_path.stat().st_size > 0


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


@pytest.mark.parametrize(
    ("layer_precision", "layer_device", "extra_kwargs"),
    [
        pytest.param([(0, trt.DataType.HALF)], None, {"fp16": True}, id="precision-half"),
        pytest.param([(0, None)], None, {}, id="precision-none-skip"),
        pytest.param(None, [(0, trt.DeviceType.GPU)], {}, id="device-gpu"),
        pytest.param(None, [(0, None)], {}, id="device-none-skip"),
    ],
)
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


def test_invalid_onnx_raises(invalid_onnx_file, output_engine_path) -> None:
    """Invalid ONNX file raises RuntimeError."""
    with pytest.raises(RuntimeError, match="Cannot parse ONNX file"):
        build_engine(invalid_onnx_file, output_engine_path, optimization_level=1)


def test_nonexistent_onnx_raises(output_engine_path, tmp_path) -> None:
    """Nonexistent ONNX path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        build_engine(
            tmp_path / "nonexistent.onnx",
            output_engine_path,
            optimization_level=1,
        )


def test_build_failure_raises(onnx_path, output_engine_path) -> None:
    """Engine build returning None raises RuntimeError."""
    # Mock the builder to return None for engine_bytes
    with patch("trtutils.builder._build.FLAGS") as mock_flags:
        # Copy real flags but force BUILD_SERIALIZED to True so we mock the right path
        for attr in dir(REAL_FLAGS):
            if not attr.startswith("_"):
                setattr(mock_flags, attr, getattr(REAL_FLAGS, attr))
        mock_flags.BUILD_SERIALIZED = True
        mock_flags.BUILD_PROGRESS = False

        # Now mock the builder.build_serialized_network to return None
        with patch("trtutils.builder._build.read_onnx") as mock_read:
            mock_network = MagicMock()
            mock_builder = MagicMock()
            mock_config = MagicMock()
            mock_parser = MagicMock()
            mock_read.return_value = (mock_network, mock_builder, mock_config, mock_parser)
            mock_builder.build_serialized_network.return_value = None
            mock_builder.create_optimization_profile.return_value = MagicMock()

            with pytest.raises(RuntimeError, match="Failed to build engine"):
                build_engine(onnx_path, output_engine_path, optimization_level=1)
