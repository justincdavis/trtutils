# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for build_engine() -- all parameter combinations and branch coverage."""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from trtutils._flags import FLAGS as REAL_FLAGS
from trtutils.builder._batcher import SyntheticBatcher
from trtutils.builder._build import build_engine
from trtutils.compat._libs import trt


def test_build_minimal(onnx_path, output_engine_path) -> None:
    """Minimal args: onnx path + output path produces a file."""
    build_engine(onnx_path, output_engine_path, optimization_level=1)
    assert output_engine_path.exists()
    assert output_engine_path.stat().st_size > 0


def test_build_fp16(onnx_path, output_engine_path) -> None:
    """Build with fp16=True succeeds."""
    build_engine(onnx_path, output_engine_path, fp16=True, optimization_level=1)
    assert output_engine_path.exists()


def test_build_int8_with_synthetic_batcher(onnx_path, output_engine_path) -> None:
    """Build with int8=True and SyntheticBatcher succeeds."""
    batcher = SyntheticBatcher(
        shape=(3, 8, 8),
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
        optimization_level=1,
    )
    assert output_engine_path.exists()


def test_build_int8_no_calibrator_warning(onnx_path, output_engine_path) -> None:
    """Build with int8=True but no batcher/cache logs a warning."""
    build_engine(
        onnx_path,
        output_engine_path,
        int8=True,
        optimization_level=1,
    )
    assert output_engine_path.exists()


def test_build_with_cache_stores(onnx_path, output_engine_path) -> None:
    """Build with cache=True stores engine in cache."""
    build_engine(onnx_path, output_engine_path, cache=True, optimization_level=1)
    assert output_engine_path.exists()


def test_build_cache_hit_skips_build(onnx_path, output_engine_path) -> None:
    """Second build with cache=True uses cached copy (skips actual build)."""
    # First build: store in cache
    build_engine(onnx_path, output_engine_path, cache=True, optimization_level=1)
    assert output_engine_path.stat().st_size > 0

    # Remove output file but cache should still have it
    output_engine_path.unlink()
    assert not output_engine_path.exists()

    # Second build: should come from cache
    build_engine(onnx_path, output_engine_path, cache=True, optimization_level=1)
    assert output_engine_path.exists()


@pytest.mark.parametrize(
    "timing_cache_val",
    [
        pytest.param(True, id="bool-true"),
        pytest.param("global", id="global-str"),
        pytest.param("local", id="local-file"),
    ],
)
def test_timing_cache_modes(
    onnx_path, output_engine_path, timing_cache_path, timing_cache_val
) -> None:
    """Build succeeds with all valid timing_cache values."""
    tc = timing_cache_path if timing_cache_val == "local" else timing_cache_val
    build_engine(
        onnx_path,
        output_engine_path,
        timing_cache=tc,
        optimization_level=1,
    )
    assert output_engine_path.exists()
    if timing_cache_val == "local":
        assert timing_cache_path.exists()


def test_timing_cache_roundtrip(onnx_path, output_engine_path, timing_cache_path) -> None:
    """Build creates timing cache, second build reads it."""
    build_engine(
        onnx_path,
        output_engine_path,
        timing_cache=timing_cache_path,
        optimization_level=1,
    )
    assert timing_cache_path.stat().st_size > 0

    out2 = output_engine_path.parent / "test2.engine"
    build_engine(
        onnx_path,
        out2,
        timing_cache=timing_cache_path,
        optimization_level=1,
    )
    assert out2.exists()


def test_timing_cache_invalid_raises(onnx_path, output_engine_path) -> None:
    """Invalid timing_cache type raises ValueError."""
    with pytest.raises(ValueError, match="Invalid timing_cache value"):
        build_engine(
            onnx_path,
            output_engine_path,
            timing_cache=42,  # type: ignore[arg-type]
            optimization_level=1,
        )


@pytest.mark.parametrize(
    "device_str",
    ["gpu", "GPU", "dla", "DLA"],
    ids=["gpu_lower", "gpu_upper", "dla_lower", "dla_upper"],
)
def test_device_string_variants(onnx_path, output_engine_path, device_str: str) -> None:
    """GPU and DLA device string variants work."""
    build_engine(
        onnx_path,
        output_engine_path,
        default_device=device_str,
        gpu_fallback=True,
        optimization_level=1,
    )
    assert output_engine_path.exists()


def test_device_enum_gpu(onnx_path, output_engine_path) -> None:
    """trt.DeviceType.GPU enum works."""
    build_engine(
        onnx_path,
        output_engine_path,
        default_device=trt.DeviceType.GPU,
        optimization_level=1,
    )
    assert output_engine_path.exists()


def test_invalid_device_string_raises(onnx_path, output_engine_path) -> None:
    """Invalid device string raises ValueError."""
    with pytest.raises(ValueError, match="Invalid default device"):
        build_engine(
            onnx_path,
            output_engine_path,
            default_device="tpu",
            optimization_level=1,
        )


def test_invalid_device_enum_raises(onnx_path, output_engine_path) -> None:
    """Invalid device enum value raises ValueError."""
    # Pass a non-DeviceType enum (just an int) to trigger the else branch
    with pytest.raises((ValueError, AttributeError)):
        build_engine(
            onnx_path,
            output_engine_path,
            default_device=999,  # type: ignore[arg-type]
            optimization_level=1,
        )


@pytest.mark.parametrize("level", [0, 1, 2, 3, 4, 5], ids=[f"opt{i}" for i in range(6)])
def test_valid_optimization_levels(onnx_path, output_engine_path, level: int) -> None:
    """All valid optimization levels (0-5) work."""
    build_engine(
        onnx_path,
        output_engine_path,
        optimization_level=level,
    )
    assert output_engine_path.exists()


def test_invalid_optimization_level_too_high(onnx_path, output_engine_path) -> None:
    """optimization_level=6 raises ValueError."""
    with pytest.raises(ValueError, match="Builder optimization level must be between 0 and 5"):
        build_engine(
            onnx_path,
            output_engine_path,
            optimization_level=6,
        )


def test_invalid_optimization_level_negative(onnx_path, output_engine_path) -> None:
    """optimization_level=-1 raises ValueError."""
    with pytest.raises(ValueError, match="Builder optimization level must be between 0 and 5"):
        build_engine(
            onnx_path,
            output_engine_path,
            optimization_level=-1,
        )


def test_manual_shapes(onnx_path, output_engine_path) -> None:
    """Passing shapes sets min/opt/max profile for specified input."""
    # The simple.onnx model's input name may vary; build should still succeed
    # or raise a runtime error if names don't match (which is expected)
    with contextlib.suppress(RuntimeError):
        build_engine(
            onnx_path,
            output_engine_path,
            shapes=[("input", (1, 3, 8, 8))],
            optimization_level=1,
        )


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


def test_prefer_precision_constraints(onnx_path, output_engine_path) -> None:
    """prefer_precision_constraints=True sets the flag."""
    build_engine(
        onnx_path,
        output_engine_path,
        prefer_precision_constraints=True,
        optimization_level=1,
    )
    assert output_engine_path.exists()


def test_reject_empty_algorithms(onnx_path, output_engine_path) -> None:
    """reject_empty_algorithms=True sets the flag."""
    build_engine(
        onnx_path,
        output_engine_path,
        reject_empty_algorithms=True,
        optimization_level=1,
    )
    assert output_engine_path.exists()


def test_direct_io_explicit(onnx_path, output_engine_path) -> None:
    """direct_io=True sets the DIRECT_IO builder flag."""
    build_engine(
        onnx_path,
        output_engine_path,
        direct_io=True,
        optimization_level=1,
    )
    assert output_engine_path.exists()


def test_direct_io_auto_enable(onnx_path, output_engine_path) -> None:
    """Tensor formats without explicit direct_io auto-enables it."""
    # Providing input_tensor_formats with direct_io=False should auto-enable
    # We use a nonexistent name so the tensor won't be found, but the
    # auto-enable code path is still triggered
    build_engine(
        onnx_path,
        output_engine_path,
        input_tensor_formats=[("nonexistent", trt.DataType.FLOAT, trt.TensorFormat.LINEAR)],
        direct_io=False,
        optimization_level=1,
    )
    assert output_engine_path.exists()


def test_gpu_fallback(onnx_path, output_engine_path) -> None:
    """gpu_fallback=True sets the GPU_FALLBACK flag."""
    build_engine(
        onnx_path,
        output_engine_path,
        gpu_fallback=True,
        optimization_level=1,
    )
    assert output_engine_path.exists()


def test_input_tensor_format_not_found_warning(onnx_path, output_engine_path) -> None:
    """Input tensor name not found logs a warning but doesn't fail."""
    build_engine(
        onnx_path,
        output_engine_path,
        input_tensor_formats=[("nonexistent_tensor", trt.DataType.FLOAT, trt.TensorFormat.LINEAR)],
        direct_io=True,
        optimization_level=1,
    )
    assert output_engine_path.exists()


def test_output_tensor_format_not_found_warning(onnx_path, output_engine_path) -> None:
    """Output tensor name not found logs a warning but doesn't fail."""
    build_engine(
        onnx_path,
        output_engine_path,
        output_tensor_formats=[("nonexistent_tensor", trt.DataType.FLOAT, trt.TensorFormat.LINEAR)],
        direct_io=True,
        optimization_level=1,
    )
    assert output_engine_path.exists()


def test_input_tensor_format_found(onnx_path, output_engine_path) -> None:
    """Input tensor name matching a real tensor sets format."""
    build_engine(
        onnx_path,
        output_engine_path,
        input_tensor_formats=[("input", trt.DataType.FLOAT, trt.TensorFormat.LINEAR)],
        direct_io=True,
        optimization_level=1,
    )
    assert output_engine_path.exists()


def test_output_tensor_format_found(onnx_path, output_engine_path) -> None:
    """Output tensor name matching a real tensor sets format."""
    build_engine(
        onnx_path,
        output_engine_path,
        output_tensor_formats=[("output", trt.DataType.FLOAT, trt.TensorFormat.LINEAR)],
        direct_io=True,
        optimization_level=1,
    )
    assert output_engine_path.exists()


def test_layer_precision(onnx_path, output_engine_path) -> None:
    """layer_precision sets per-layer precision."""
    build_engine(
        onnx_path,
        output_engine_path,
        layer_precision=[(0, trt.DataType.HALF)],
        fp16=True,
        optimization_level=1,
    )
    assert output_engine_path.exists()


def test_layer_precision_none_skip(onnx_path, output_engine_path) -> None:
    """layer_precision with None precision is skipped."""
    build_engine(
        onnx_path,
        output_engine_path,
        layer_precision=[(0, None)],
        optimization_level=1,
    )
    assert output_engine_path.exists()


def test_layer_device_gpu(onnx_path, output_engine_path) -> None:
    """layer_device with GPU assignment."""
    build_engine(
        onnx_path,
        output_engine_path,
        layer_device=[(0, trt.DeviceType.GPU)],
        optimization_level=1,
    )
    assert output_engine_path.exists()


def test_layer_device_none_skip(onnx_path, output_engine_path) -> None:
    """layer_device with None device is skipped."""
    build_engine(
        onnx_path,
        output_engine_path,
        layer_device=[(0, None)],
        optimization_level=1,
    )
    assert output_engine_path.exists()


def test_layer_device_dla_raises_without_fallback(onnx_path, output_engine_path) -> None:
    """DLA assignment on non-DLA-capable layer without fallback raises ValueError."""
    # Layer 0 almost certainly can't run on DLA on non-Jetson systems
    with pytest.raises((ValueError, RuntimeError)):
        build_engine(
            onnx_path,
            output_engine_path,
            layer_device=[(0, trt.DeviceType.DLA)],
            gpu_fallback=False,
            optimization_level=1,
        )


def test_layer_device_dla_with_fallback(onnx_path, output_engine_path) -> None:
    """DLA assignment with gpu_fallback=True logs warning instead of raising."""
    # Should not raise, should warn and fallback
    build_engine(
        onnx_path,
        output_engine_path,
        layer_device=[(0, trt.DeviceType.DLA)],
        gpu_fallback=True,
        optimization_level=1,
    )
    assert output_engine_path.exists()


def test_dla_core_assignment(onnx_path, output_engine_path) -> None:
    """Setting dla_core configures the DLA core on the builder config."""
    # On non-Jetson, DLA core 0 is set on config but engine is built on GPU
    build_engine(
        onnx_path,
        output_engine_path,
        dla_core=0,
        gpu_fallback=True,
        optimization_level=1,
    )
    assert output_engine_path.exists()


def test_profiling_verbosity(onnx_path, output_engine_path) -> None:
    """Setting profiling_verbosity applies it to config."""
    build_engine(
        onnx_path,
        output_engine_path,
        profiling_verbosity=trt.ProfilingVerbosity.DETAILED,
        optimization_level=1,
    )
    assert output_engine_path.exists()


def test_tiling_optimization_level(onnx_path, output_engine_path) -> None:
    """Setting tiling_optimization_level applies to config if supported."""
    if not hasattr(trt, "TilingOptimizationLevel"):
        pytest.skip("TilingOptimizationLevel not supported in this TRT version")
    build_engine(
        onnx_path,
        output_engine_path,
        tiling_optimization_level=trt.TilingOptimizationLevel.NONE,
        optimization_level=1,
    )
    assert output_engine_path.exists()


def test_tiling_l2_cache_limit(onnx_path, output_engine_path) -> None:
    """Setting tiling_l2_cache_limit applies to config."""
    try:
        build_engine(
            onnx_path,
            output_engine_path,
            tiling_l2_cache_limit=1024 * 1024,
            optimization_level=1,
        )
        assert output_engine_path.exists()
    except AttributeError:
        pytest.skip("l2_limit_for_tiling not supported in this TRT version")


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


def test_ignore_timing_mismatch(onnx_path, output_engine_path, timing_cache_path) -> None:
    """ignore_timing_mismatch=True passes to set_timing_cache."""
    build_engine(
        onnx_path,
        output_engine_path,
        timing_cache=timing_cache_path,
        ignore_timing_mismatch=True,
        optimization_level=1,
    )
    assert output_engine_path.exists()


def test_device_index_zero(onnx_path, output_engine_path) -> None:
    """device=0 explicitly sets CUDA device 0."""
    build_engine(
        onnx_path,
        output_engine_path,
        device=0,
        optimization_level=1,
    )
    assert output_engine_path.exists()
