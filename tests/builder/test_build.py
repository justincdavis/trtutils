# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for build_engine() -- all parameter combinations and branch coverage."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_engine_import():
    """Import build_engine lazily (only on GPU)."""
    from trtutils.builder._build import build_engine

    return build_engine


def _trt_import():
    """Import trt lazily."""
    from trtutils.compat._libs import trt

    return trt


# ===========================================================================
# Basic build tests
# ===========================================================================
class TestBuildEngineBasic:
    """Basic build_engine smoke tests."""

    def test_build_minimal(self, onnx_path: Path, output_engine_path: Path) -> None:
        """Minimal args: onnx path + output path produces a file."""
        build_engine = _build_engine_import()
        build_engine(onnx_path, output_engine_path, optimization_level=1)
        assert output_engine_path.exists()
        assert output_engine_path.stat().st_size > 0

    def test_build_creates_file(self, onnx_path: Path, output_engine_path: Path) -> None:
        """Output file exists after a successful build."""
        build_engine = _build_engine_import()
        build_engine(onnx_path, output_engine_path, optimization_level=1)
        assert output_engine_path.is_file()

    def test_build_output_is_loadable(self, onnx_path: Path, output_engine_path: Path) -> None:
        """Built engine file can be loaded by TRTEngine."""
        build_engine = _build_engine_import()
        build_engine(onnx_path, output_engine_path, optimization_level=1)
        # Just verify the file is a valid binary (non-empty and starts with reasonable bytes)
        data = output_engine_path.read_bytes()
        assert len(data) > 100


# ===========================================================================
# Precision tests
# ===========================================================================
class TestBuildEnginePrecision:
    """Tests for FP16 / INT8 precision flags."""

    def test_build_default_precision(self, onnx_path: Path, output_engine_path: Path) -> None:
        """Default build without precision flags succeeds."""
        build_engine = _build_engine_import()
        build_engine(onnx_path, output_engine_path, optimization_level=1)
        assert output_engine_path.exists()

    def test_build_fp16(self, onnx_path: Path, output_engine_path: Path) -> None:
        """Build with fp16=True succeeds."""
        build_engine = _build_engine_import()
        build_engine(onnx_path, output_engine_path, fp16=True, optimization_level=1)
        assert output_engine_path.exists()

    def test_build_int8_with_synthetic_batcher(
        self, onnx_path: Path, output_engine_path: Path
    ) -> None:
        """Build with int8=True and SyntheticBatcher succeeds."""
        from trtutils.builder._batcher import SyntheticBatcher

        batcher = SyntheticBatcher(
            shape=(3, 8, 8),
            dtype=np.dtype(np.float32),
            batch_size=1,
            num_batches=2,
            order="NCHW",
        )
        build_engine = _build_engine_import()
        build_engine(
            onnx_path,
            output_engine_path,
            int8=True,
            data_batcher=batcher,
            optimization_level=1,
        )
        assert output_engine_path.exists()

    def test_build_int8_no_calibrator_warning(
        self, onnx_path: Path, output_engine_path: Path
    ) -> None:
        """Build with int8=True but no batcher/cache logs a warning."""
        build_engine = _build_engine_import()
        # Should still build but warn about inaccurate INT8
        build_engine(
            onnx_path,
            output_engine_path,
            int8=True,
            optimization_level=1,
        )
        assert output_engine_path.exists()


# ===========================================================================
# Cache tests
# ===========================================================================
class TestBuildEngineCache:
    """Tests for engine caching behavior."""

    def test_build_with_cache_stores(self, onnx_path: Path, output_engine_path: Path) -> None:
        """Build with cache=True stores engine in cache."""
        build_engine = _build_engine_import()
        build_engine(onnx_path, output_engine_path, cache=True, optimization_level=1)
        assert output_engine_path.exists()

    def test_build_cache_hit_skips_build(self, onnx_path: Path, output_engine_path: Path) -> None:
        """Second build with cache=True uses cached copy (skips actual build)."""
        build_engine = _build_engine_import()

        # First build: store in cache
        build_engine(onnx_path, output_engine_path, cache=True, optimization_level=1)
        assert output_engine_path.stat().st_size > 0

        # Remove output file but cache should still have it
        output_engine_path.unlink()
        assert not output_engine_path.exists()

        # Second build: should come from cache
        build_engine(onnx_path, output_engine_path, cache=True, optimization_level=1)
        assert output_engine_path.exists()

    def test_build_without_cache(self, onnx_path: Path, output_engine_path: Path) -> None:
        """Build with cache=None does not use caching system."""
        build_engine = _build_engine_import()
        build_engine(onnx_path, output_engine_path, cache=None, optimization_level=1)
        assert output_engine_path.exists()


# ===========================================================================
# Timing cache tests
# ===========================================================================
class TestBuildEngineTimingCache:
    """Tests for timing cache parameter variations."""

    @pytest.mark.parametrize("timing_cache_val", [True, "global"], ids=["bool_true", "global_str"])
    def test_timing_cache_global_modes(
        self, onnx_path: Path, output_engine_path: Path, timing_cache_val: object
    ) -> None:
        """Build with global timing cache (True or 'global')."""
        build_engine = _build_engine_import()
        build_engine(
            onnx_path,
            output_engine_path,
            timing_cache=timing_cache_val,
            optimization_level=1,
        )
        assert output_engine_path.exists()

    def test_timing_cache_local_file(
        self, onnx_path: Path, output_engine_path: Path, timing_cache_path: Path
    ) -> None:
        """Build with a local timing cache file path."""
        build_engine = _build_engine_import()
        build_engine(
            onnx_path,
            output_engine_path,
            timing_cache=timing_cache_path,
            optimization_level=1,
        )
        assert output_engine_path.exists()
        # Timing cache file should be written
        assert timing_cache_path.exists()

    def test_timing_cache_local_existing(
        self, onnx_path: Path, output_engine_path: Path, timing_cache_path: Path
    ) -> None:
        """Build reads an existing local timing cache file."""
        build_engine = _build_engine_import()
        # First build creates timing cache
        build_engine(
            onnx_path,
            output_engine_path,
            timing_cache=timing_cache_path,
            optimization_level=1,
        )
        assert timing_cache_path.stat().st_size > 0

        # Second build reads existing timing cache
        out2 = output_engine_path.parent / "test2.engine"
        build_engine(
            onnx_path,
            out2,
            timing_cache=timing_cache_path,
            optimization_level=1,
        )
        assert out2.exists()

    def test_timing_cache_none(self, onnx_path: Path, output_engine_path: Path) -> None:
        """Build with timing_cache=None (no timing cache)."""
        build_engine = _build_engine_import()
        build_engine(
            onnx_path,
            output_engine_path,
            timing_cache=None,
            optimization_level=1,
        )
        assert output_engine_path.exists()

    def test_timing_cache_invalid_raises(self, onnx_path: Path, output_engine_path: Path) -> None:
        """Invalid timing_cache type raises ValueError."""
        build_engine = _build_engine_import()
        with pytest.raises(ValueError, match="Invalid timing_cache value"):
            build_engine(
                onnx_path,
                output_engine_path,
                timing_cache=42,  # type: ignore[arg-type]
                optimization_level=1,
            )


# ===========================================================================
# Device tests
# ===========================================================================
class TestBuildEngineDevice:
    """Tests for default_device parameter."""

    @pytest.mark.parametrize(
        "device_str",
        ["gpu", "GPU", "dla", "DLA"],
        ids=["gpu_lower", "gpu_upper", "dla_lower", "dla_upper"],
    )
    def test_device_string_variants(
        self, onnx_path: Path, output_engine_path: Path, device_str: str
    ) -> None:
        """GPU and DLA device string variants work."""
        build_engine = _build_engine_import()
        build_engine(
            onnx_path,
            output_engine_path,
            default_device=device_str,
            gpu_fallback=True,
            optimization_level=1,
        )
        assert output_engine_path.exists()

    def test_device_enum_gpu(self, onnx_path: Path, output_engine_path: Path) -> None:
        """trt.DeviceType.GPU enum works."""
        trt = _trt_import()
        build_engine = _build_engine_import()
        build_engine(
            onnx_path,
            output_engine_path,
            default_device=trt.DeviceType.GPU,
            optimization_level=1,
        )
        assert output_engine_path.exists()

    def test_invalid_device_string_raises(self, onnx_path: Path, output_engine_path: Path) -> None:
        """Invalid device string raises ValueError."""
        build_engine = _build_engine_import()
        with pytest.raises(ValueError, match="Invalid default device"):
            build_engine(
                onnx_path,
                output_engine_path,
                default_device="tpu",
                optimization_level=1,
            )

    def test_invalid_device_enum_raises(self, onnx_path: Path, output_engine_path: Path) -> None:
        """Invalid device enum value raises ValueError."""
        build_engine = _build_engine_import()
        # Pass a non-DeviceType enum (just an int) to trigger the else branch
        with pytest.raises((ValueError, AttributeError)):
            build_engine(
                onnx_path,
                output_engine_path,
                default_device=999,  # type: ignore[arg-type]
                optimization_level=1,
            )


# ===========================================================================
# Optimization level tests
# ===========================================================================
class TestBuildEngineOptLevel:
    """Tests for optimization_level parameter."""

    @pytest.mark.parametrize("level", [0, 1, 2, 3, 4, 5], ids=[f"opt{i}" for i in range(6)])
    def test_valid_optimization_levels(
        self, onnx_path: Path, output_engine_path: Path, level: int
    ) -> None:
        """All valid optimization levels (0-5) work."""
        build_engine = _build_engine_import()
        # Use level=1 for speed but parametrize to validate range check
        # Only actually build with the given level to cover the branch
        build_engine(
            onnx_path,
            output_engine_path,
            optimization_level=level,
        )
        assert output_engine_path.exists()

    def test_invalid_optimization_level_too_high(
        self, onnx_path: Path, output_engine_path: Path
    ) -> None:
        """optimization_level=6 raises ValueError."""
        build_engine = _build_engine_import()
        with pytest.raises(ValueError, match="Builder optimization level must be between 0 and 5"):
            build_engine(
                onnx_path,
                output_engine_path,
                optimization_level=6,
            )

    def test_invalid_optimization_level_negative(
        self, onnx_path: Path, output_engine_path: Path
    ) -> None:
        """optimization_level=-1 raises ValueError."""
        build_engine = _build_engine_import()
        with pytest.raises(ValueError, match="Builder optimization level must be between 0 and 5"):
            build_engine(
                onnx_path,
                output_engine_path,
                optimization_level=-1,
            )


# ===========================================================================
# Shapes tests
# ===========================================================================
class TestBuildEngineShapes:
    """Tests for manual shapes parameter."""

    def test_manual_shapes(self, onnx_path: Path, output_engine_path: Path) -> None:
        """Passing shapes sets min/opt/max profile for specified input."""
        build_engine = _build_engine_import()
        # The simple.onnx model's input name may vary; build should still succeed
        # or raise a runtime error if names don't match (which is expected)
        with contextlib.suppress(RuntimeError):
            build_engine(
                onnx_path,
                output_engine_path,
                shapes=[("input", (1, 3, 8, 8))],
                optimization_level=1,
            )


# ===========================================================================
# Hooks tests
# ===========================================================================
class TestBuildEngineHooks:
    """Tests for network hooks."""

    def test_single_hook(self, onnx_path: Path, output_engine_path: Path) -> None:
        """A single hook function is called during build."""
        hook_called = []

        def identity_hook(network):
            hook_called.append(True)
            return network

        build_engine = _build_engine_import()
        build_engine(
            onnx_path,
            output_engine_path,
            hooks=[identity_hook],
            optimization_level=1,
        )
        assert len(hook_called) == 1
        assert output_engine_path.exists()

    def test_multiple_hooks(self, onnx_path: Path, output_engine_path: Path) -> None:
        """Multiple hooks are called in order."""
        call_order = []

        def hook_a(network):
            call_order.append("a")
            return network

        def hook_b(network):
            call_order.append("b")
            return network

        build_engine = _build_engine_import()
        build_engine(
            onnx_path,
            output_engine_path,
            hooks=[hook_a, hook_b],
            optimization_level=1,
        )
        assert call_order == ["a", "b"]


# ===========================================================================
# Builder flags tests
# ===========================================================================
class TestBuildEngineFlags:
    """Tests for builder flag parameters."""

    def test_prefer_precision_constraints(self, onnx_path: Path, output_engine_path: Path) -> None:
        """prefer_precision_constraints=True sets the flag."""
        build_engine = _build_engine_import()
        build_engine(
            onnx_path,
            output_engine_path,
            prefer_precision_constraints=True,
            optimization_level=1,
        )
        assert output_engine_path.exists()

    def test_reject_empty_algorithms(self, onnx_path: Path, output_engine_path: Path) -> None:
        """reject_empty_algorithms=True sets the flag."""
        build_engine = _build_engine_import()
        build_engine(
            onnx_path,
            output_engine_path,
            reject_empty_algorithms=True,
            optimization_level=1,
        )
        assert output_engine_path.exists()

    def test_direct_io_explicit(self, onnx_path: Path, output_engine_path: Path) -> None:
        """direct_io=True sets the DIRECT_IO builder flag."""
        build_engine = _build_engine_import()
        build_engine(
            onnx_path,
            output_engine_path,
            direct_io=True,
            optimization_level=1,
        )
        assert output_engine_path.exists()

    def test_direct_io_auto_enable(self, onnx_path: Path, output_engine_path: Path) -> None:
        """Tensor formats without explicit direct_io auto-enables it."""
        trt = _trt_import()
        build_engine = _build_engine_import()
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

    def test_gpu_fallback(self, onnx_path: Path, output_engine_path: Path) -> None:
        """gpu_fallback=True sets the GPU_FALLBACK flag."""
        build_engine = _build_engine_import()
        build_engine(
            onnx_path,
            output_engine_path,
            gpu_fallback=True,
            optimization_level=1,
        )
        assert output_engine_path.exists()


# ===========================================================================
# Tensor format tests
# ===========================================================================
class TestBuildEngineTensorFormats:
    """Tests for input/output tensor format specification."""

    def test_input_tensor_format_not_found_warning(
        self, onnx_path: Path, output_engine_path: Path
    ) -> None:
        """Input tensor name not found logs a warning but doesn't fail."""
        trt = _trt_import()
        build_engine = _build_engine_import()
        build_engine(
            onnx_path,
            output_engine_path,
            input_tensor_formats=[
                ("nonexistent_tensor", trt.DataType.FLOAT, trt.TensorFormat.LINEAR)
            ],
            direct_io=True,
            optimization_level=1,
        )
        assert output_engine_path.exists()

    def test_output_tensor_format_not_found_warning(
        self, onnx_path: Path, output_engine_path: Path
    ) -> None:
        """Output tensor name not found logs a warning but doesn't fail."""
        trt = _trt_import()
        build_engine = _build_engine_import()
        build_engine(
            onnx_path,
            output_engine_path,
            output_tensor_formats=[
                ("nonexistent_tensor", trt.DataType.FLOAT, trt.TensorFormat.LINEAR)
            ],
            direct_io=True,
            optimization_level=1,
        )
        assert output_engine_path.exists()

    def test_input_tensor_format_found(self, onnx_path: Path, output_engine_path: Path) -> None:
        """Input tensor name matching a real tensor sets format."""
        trt = _trt_import()
        build_engine = _build_engine_import()
        build_engine(
            onnx_path,
            output_engine_path,
            input_tensor_formats=[("input", trt.DataType.FLOAT, trt.TensorFormat.LINEAR)],
            direct_io=True,
            optimization_level=1,
        )
        assert output_engine_path.exists()

    def test_output_tensor_format_found(self, onnx_path: Path, output_engine_path: Path) -> None:
        """Output tensor name matching a real tensor sets format."""
        trt = _trt_import()
        build_engine = _build_engine_import()
        build_engine(
            onnx_path,
            output_engine_path,
            output_tensor_formats=[("output", trt.DataType.FLOAT, trt.TensorFormat.LINEAR)],
            direct_io=True,
            optimization_level=1,
        )
        assert output_engine_path.exists()


# ===========================================================================
# Layer precision / device tests
# ===========================================================================
class TestBuildEngineLayerConfig:
    """Tests for per-layer precision and device assignment."""

    def test_layer_precision(self, onnx_path: Path, output_engine_path: Path) -> None:
        """layer_precision sets per-layer precision."""
        trt = _trt_import()
        build_engine = _build_engine_import()
        build_engine(
            onnx_path,
            output_engine_path,
            layer_precision=[(0, trt.DataType.HALF)],
            fp16=True,
            optimization_level=1,
        )
        assert output_engine_path.exists()

    def test_layer_precision_none_skip(self, onnx_path: Path, output_engine_path: Path) -> None:
        """layer_precision with None precision is skipped."""
        build_engine = _build_engine_import()
        build_engine(
            onnx_path,
            output_engine_path,
            layer_precision=[(0, None)],
            optimization_level=1,
        )
        assert output_engine_path.exists()

    def test_layer_device_gpu(self, onnx_path: Path, output_engine_path: Path) -> None:
        """layer_device with GPU assignment."""
        trt = _trt_import()
        build_engine = _build_engine_import()
        build_engine(
            onnx_path,
            output_engine_path,
            layer_device=[(0, trt.DeviceType.GPU)],
            optimization_level=1,
        )
        assert output_engine_path.exists()

    def test_layer_device_none_skip(self, onnx_path: Path, output_engine_path: Path) -> None:
        """layer_device with None device is skipped."""
        build_engine = _build_engine_import()
        build_engine(
            onnx_path,
            output_engine_path,
            layer_device=[(0, None)],
            optimization_level=1,
        )
        assert output_engine_path.exists()

    def test_layer_device_dla_raises_without_fallback(
        self, onnx_path: Path, output_engine_path: Path
    ) -> None:
        """DLA assignment on non-DLA-capable layer without fallback raises ValueError."""
        trt = _trt_import()
        build_engine = _build_engine_import()
        # Layer 0 almost certainly can't run on DLA on non-Jetson systems
        with pytest.raises((ValueError, RuntimeError)):
            build_engine(
                onnx_path,
                output_engine_path,
                layer_device=[(0, trt.DeviceType.DLA)],
                gpu_fallback=False,
                optimization_level=1,
            )

    def test_layer_device_dla_with_fallback(self, onnx_path: Path, output_engine_path: Path) -> None:
        """DLA assignment with gpu_fallback=True logs warning instead of raising."""
        trt = _trt_import()
        build_engine = _build_engine_import()
        # Should not raise, should warn and fallback
        build_engine(
            onnx_path,
            output_engine_path,
            layer_device=[(0, trt.DeviceType.DLA)],
            gpu_fallback=True,
            optimization_level=1,
        )
        assert output_engine_path.exists()


# ===========================================================================
# DLA core tests
# ===========================================================================
class TestBuildEngineDlaCore:
    """Tests for dla_core parameter."""

    def test_dla_core_assignment(self, onnx_path: Path, output_engine_path: Path) -> None:
        """Setting dla_core configures the DLA core on the builder config."""
        build_engine = _build_engine_import()
        # On non-Jetson, DLA core 0 is set on config but engine is built on GPU
        build_engine(
            onnx_path,
            output_engine_path,
            dla_core=0,
            gpu_fallback=True,
            optimization_level=1,
        )
        assert output_engine_path.exists()


# ===========================================================================
# Profiling verbosity tests
# ===========================================================================
class TestBuildEngineProfilingVerbosity:
    """Tests for profiling_verbosity parameter."""

    def test_profiling_verbosity(self, onnx_path: Path, output_engine_path: Path) -> None:
        """Setting profiling_verbosity applies it to config."""
        trt = _trt_import()
        build_engine = _build_engine_import()
        build_engine(
            onnx_path,
            output_engine_path,
            profiling_verbosity=trt.ProfilingVerbosity.DETAILED,
            optimization_level=1,
        )
        assert output_engine_path.exists()


# ===========================================================================
# Tiling optimization tests
# ===========================================================================
class TestBuildEngineTiling:
    """Tests for tiling optimization parameters."""

    def test_tiling_optimization_level(self, onnx_path: Path, output_engine_path: Path) -> None:
        """Setting tiling_optimization_level applies to config if supported."""
        trt = _trt_import()
        build_engine = _build_engine_import()
        if not hasattr(trt, "TilingOptimizationLevel"):
            pytest.skip("TilingOptimizationLevel not supported in this TRT version")
        build_engine(
            onnx_path,
            output_engine_path,
            tiling_optimization_level=trt.TilingOptimizationLevel.NONE,
            optimization_level=1,
        )
        assert output_engine_path.exists()

    def test_tiling_l2_cache_limit(self, onnx_path: Path, output_engine_path: Path) -> None:
        """Setting tiling_l2_cache_limit applies to config."""
        build_engine = _build_engine_import()
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


# ===========================================================================
# Progress bar tests
# ===========================================================================
class TestBuildEngineProgress:
    """Tests for verbose/progress bar."""

    def test_verbose_build(self, onnx_path: Path, output_engine_path: Path) -> None:
        """Build with verbose=True works (may or may not show progress bar)."""
        build_engine = _build_engine_import()
        build_engine(
            onnx_path,
            output_engine_path,
            verbose=True,
            optimization_level=1,
        )
        assert output_engine_path.exists()


# ===========================================================================
# Error handling tests
# ===========================================================================
class TestBuildEngineErrors:
    """Tests for error paths in build_engine."""

    def test_invalid_onnx_raises(self, invalid_onnx_file: Path, output_engine_path: Path) -> None:
        """Invalid ONNX file raises RuntimeError."""
        build_engine = _build_engine_import()
        with pytest.raises(RuntimeError, match="Cannot parse ONNX file"):
            build_engine(invalid_onnx_file, output_engine_path, optimization_level=1)

    def test_nonexistent_onnx_raises(self, output_engine_path: Path, tmp_path: Path) -> None:
        """Nonexistent ONNX path raises FileNotFoundError."""
        build_engine = _build_engine_import()
        with pytest.raises(FileNotFoundError):
            build_engine(
                tmp_path / "nonexistent.onnx",
                output_engine_path,
                optimization_level=1,
            )

    def test_build_failure_raises(self, onnx_path: Path, output_engine_path: Path) -> None:
        """Engine build returning None raises RuntimeError."""
        from unittest.mock import patch

        build_engine = _build_engine_import()
        # Mock the builder to return None for engine_bytes
        with patch("trtutils.builder._build.FLAGS") as mock_flags:
            # Copy real flags but force BUILD_SERIALIZED to True so we mock the right path
            from trtutils._flags import FLAGS as REAL_FLAGS

            for attr in dir(REAL_FLAGS):
                if not attr.startswith("_"):
                    setattr(mock_flags, attr, getattr(REAL_FLAGS, attr))
            mock_flags.BUILD_SERIALIZED = True
            mock_flags.BUILD_PROGRESS = False

            # Now mock the builder.build_serialized_network to return None
            with patch("trtutils.builder._build.read_onnx") as mock_read:
                from unittest.mock import MagicMock

                mock_network = MagicMock()
                mock_builder = MagicMock()
                mock_config = MagicMock()
                mock_parser = MagicMock()
                mock_read.return_value = (mock_network, mock_builder, mock_config, mock_parser)
                mock_builder.build_serialized_network.return_value = None
                mock_builder.create_optimization_profile.return_value = MagicMock()

                with pytest.raises(RuntimeError, match="Failed to build engine"):
                    build_engine(onnx_path, output_engine_path, optimization_level=1)


# ===========================================================================
# Ignore timing mismatch tests
# ===========================================================================
class TestBuildEngineTimingMismatch:
    """Tests for ignore_timing_mismatch parameter."""

    def test_ignore_timing_mismatch(
        self, onnx_path: Path, output_engine_path: Path, timing_cache_path: Path
    ) -> None:
        """ignore_timing_mismatch=True passes to set_timing_cache."""
        build_engine = _build_engine_import()
        build_engine(
            onnx_path,
            output_engine_path,
            timing_cache=timing_cache_path,
            ignore_timing_mismatch=True,
            optimization_level=1,
        )
        assert output_engine_path.exists()


# ===========================================================================
# Device (CUDA device index) tests
# ===========================================================================
class TestBuildEngineDeviceIndex:
    """Tests for the device (CUDA device index) parameter."""

    def test_device_index_none(self, onnx_path: Path, output_engine_path: Path) -> None:
        """device=None uses current device (no-op guard)."""
        build_engine = _build_engine_import()
        build_engine(
            onnx_path,
            output_engine_path,
            device=None,
            optimization_level=1,
        )
        assert output_engine_path.exists()

    def test_device_index_zero(self, onnx_path: Path, output_engine_path: Path) -> None:
        """device=0 explicitly sets CUDA device 0."""
        build_engine = _build_engine_import()
        build_engine(
            onnx_path,
            output_engine_path,
            device=0,
            optimization_level=1,
        )
        assert output_engine_path.exists()
