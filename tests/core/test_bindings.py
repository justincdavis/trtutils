# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/core/_bindings.py -- Binding dataclass and allocation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Binding dataclass tests
# ---------------------------------------------------------------------------
@pytest.mark.gpu
class TestBindingDataclass:
    """Tests for the Binding dataclass fields and free() method."""

    def test_binding_fields(self) -> None:
        """Binding has all expected dataclass fields."""
        from trtutils.core._bindings import create_binding

        arr = np.zeros((1, 3, 224, 224), dtype=np.float32)
        binding = create_binding(arr)

        assert hasattr(binding, "index")
        assert hasattr(binding, "name")
        assert hasattr(binding, "dtype")
        assert hasattr(binding, "shape")
        assert hasattr(binding, "is_input")
        assert hasattr(binding, "allocation")
        assert hasattr(binding, "host_allocation")
        assert hasattr(binding, "tensor_format")
        assert hasattr(binding, "pagelocked_mem")
        assert hasattr(binding, "unified_mem")
        binding.free()

    def test_binding_is_dataclass(self) -> None:
        """Binding is a dataclass instance."""
        import dataclasses

        from trtutils.core._bindings import create_binding

        arr = np.zeros((2, 2), dtype=np.float32)
        binding = create_binding(arr)
        assert dataclasses.is_dataclass(binding)
        binding.free()

    def test_binding_free(self) -> None:
        """free() deallocates memory without error."""
        from trtutils.core._bindings import create_binding

        arr = np.zeros((10,), dtype=np.float32)
        binding = create_binding(arr)
        binding.free()  # Should not raise

    def test_binding_free_idempotent(self) -> None:
        """Calling free() multiple times does not crash (via __del__)."""
        from trtutils.core._bindings import create_binding

        arr = np.zeros((10,), dtype=np.float32)
        binding = create_binding(arr)
        binding.free()
        # The __del__ will call free() again and suppress RuntimeError
        del binding  # Should not crash

    def test_binding_shape_matches(self) -> None:
        """Binding shape matches the input array shape."""
        from trtutils.core._bindings import create_binding

        shape = (1, 3, 64, 64)
        arr = np.zeros(shape, dtype=np.float32)
        binding = create_binding(arr)
        assert binding.shape == list(shape)
        binding.free()

    def test_binding_dtype_matches(self) -> None:
        """Binding dtype matches the input array dtype."""
        from trtutils.core._bindings import create_binding

        arr = np.zeros((4, 4), dtype=np.float16)
        binding = create_binding(arr)
        assert binding.dtype == np.float16
        binding.free()


# ---------------------------------------------------------------------------
# create_binding tests
# ---------------------------------------------------------------------------
@pytest.mark.gpu
class TestCreateBinding:
    """Tests for create_binding()."""

    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(np.float32, id="float32"),
            pytest.param(np.float16, id="float16"),
            pytest.param(np.int32, id="int32"),
        ],
    )
    def test_create_binding_dtypes(self, dtype) -> None:
        """create_binding works with various numpy dtypes."""
        from trtutils.core._bindings import create_binding

        arr = np.zeros((8,), dtype=dtype)
        binding = create_binding(arr)
        assert binding.dtype == dtype
        assert binding.allocation != 0
        assert isinstance(binding.host_allocation, np.ndarray)
        binding.free()

    def test_create_binding_default_pagelocked(self) -> None:
        """Defaults to pagelocked memory (pagelocked_mem=None -> True)."""
        from trtutils.core._bindings import create_binding

        arr = np.zeros((4,), dtype=np.float32)
        binding = create_binding(arr)
        assert binding.pagelocked_mem is True
        binding.free()

    def test_create_binding_no_pagelocked(self) -> None:
        """pagelocked_mem=False uses regular numpy arrays for host."""
        from trtutils.core._bindings import create_binding

        arr = np.zeros((4,), dtype=np.float32)
        binding = create_binding(arr, pagelocked_mem=False)
        assert binding.pagelocked_mem is False
        assert isinstance(binding.host_allocation, np.ndarray)
        binding.free()

    def test_create_binding_pagelocked_explicit(self) -> None:
        """pagelocked_mem=True uses page-locked memory."""
        from trtutils.core._bindings import create_binding

        arr = np.zeros((4,), dtype=np.float32)
        binding = create_binding(arr, pagelocked_mem=True)
        assert binding.pagelocked_mem is True
        binding.free()

    def test_create_binding_unified_memory(self) -> None:
        """pagelocked=True, unified=True uses the unified memory path."""
        from trtutils.core._bindings import create_binding

        arr = np.zeros((4,), dtype=np.float32)
        binding = create_binding(arr, pagelocked_mem=True, unified_mem=True)
        assert binding.unified_mem is True
        assert binding.allocation != 0
        binding.free()

    def test_create_binding_unified_memory_frees_host_only(self) -> None:
        """Mapped unified host allocations should not be freed with cudaFree."""
        from trtutils.core import _bindings

        arr = np.zeros((4,), dtype=np.float32)
        binding = _bindings.create_binding(arr, pagelocked_mem=True, unified_mem=True)

        with patch.object(
            _bindings.cudart, "cudaFree", wraps=_bindings.cudart.cudaFree
        ) as cuda_free, patch.object(
            _bindings.cudart,
            "cudaFreeHost",
            wraps=_bindings.cudart.cudaFreeHost,
        ) as cuda_free_host:
            binding.free()

        cuda_free.assert_not_called()
        cuda_free_host.assert_called_once()

    def test_create_binding_with_array_data(self) -> None:
        """use_array_data=True copies data from the input array."""
        from trtutils.core._bindings import create_binding

        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        binding = create_binding(arr, use_array_data=True)
        np.testing.assert_array_equal(binding.host_allocation, arr)
        binding.free()

    def test_create_binding_without_array_data(self) -> None:
        """use_array_data=None does not copy input data."""
        from trtutils.core._bindings import create_binding

        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        binding = create_binding(arr)
        # Host allocation should be zeros (data not copied)
        np.testing.assert_array_equal(binding.host_allocation, np.zeros_like(arr))
        binding.free()

    def test_create_binding_name_and_id(self) -> None:
        """create_binding stores the given name and bind_id."""
        from trtutils.core._bindings import create_binding

        arr = np.zeros((2,), dtype=np.float32)
        binding = create_binding(arr, bind_id=5, name="test_tensor")
        assert binding.index == 5
        assert binding.name == "test_tensor"
        binding.free()

    def test_create_binding_is_input(self) -> None:
        """create_binding stores the is_input flag."""
        from trtutils.core._bindings import create_binding

        arr = np.zeros((2,), dtype=np.float32)
        binding_input = create_binding(arr, is_input=True)
        binding_output = create_binding(arr, is_input=False)
        assert binding_input.is_input is True
        assert binding_output.is_input is False
        binding_input.free()
        binding_output.free()

    def test_create_binding_multidim_shape(self) -> None:
        """create_binding preserves multi-dimensional shapes."""
        from trtutils.core._bindings import create_binding

        arr = np.zeros((1, 3, 224, 224), dtype=np.float32)
        binding = create_binding(arr)
        assert binding.shape == [1, 3, 224, 224]
        assert binding.host_allocation.shape == (1, 3, 224, 224)
        binding.free()


# ---------------------------------------------------------------------------
# allocate_bindings tests (requires a real TensorRT engine)
# ---------------------------------------------------------------------------
@pytest.mark.gpu
class TestAllocateBindings:
    """Tests for allocate_bindings() using a real TensorRT engine."""

    @pytest.mark.parametrize(
        ("pagelocked", "unified"),
        [
            pytest.param(True, False, id="pagelocked"),
            pytest.param(False, False, id="default"),
            pytest.param(True, True, id="unified"),
        ],
    )
    def test_allocate_returns_io_bindings(
        self, simple_engine_path, pagelocked, unified: object
    ) -> None:
        """allocate_bindings returns (inputs, outputs, allocations)."""
        from trtutils.core._bindings import allocate_bindings
        from trtutils.core._engine import create_engine
        from trtutils.core._stream import destroy_stream

        engine, context, _logger, stream = create_engine(simple_engine_path)
        try:
            inputs, outputs, allocations = allocate_bindings(
                engine,
                context,
                pagelocked_mem=pagelocked,
                unified_mem=unified,
            )

            assert isinstance(inputs, list)
            assert isinstance(outputs, list)
            assert isinstance(allocations, list)
            assert len(inputs) > 0
            assert len(outputs) > 0
            assert len(allocations) > 0

            for b in inputs + outputs:
                b.free()
        finally:
            destroy_stream(stream)

    def test_bindings_count_matches_engine(self, simple_engine_path) -> None:
        """Number of input+output bindings matches engine tensor count."""
        from trtutils._flags import FLAGS
        from trtutils.core._bindings import allocate_bindings
        from trtutils.core._engine import create_engine
        from trtutils.core._stream import destroy_stream

        engine, context, _logger, stream = create_engine(simple_engine_path)
        try:
            inputs, outputs, allocations = allocate_bindings(engine, context)

            expected = engine.num_io_tensors if FLAGS.TRT_10 else engine.num_bindings

            assert len(inputs) + len(outputs) == expected
            assert len(allocations) == expected

            for b in inputs + outputs:
                b.free()
        finally:
            destroy_stream(stream)

    def test_bindings_have_device_allocation(self, simple_engine_path) -> None:
        """Each binding has a non-zero device allocation pointer."""
        from trtutils.core._bindings import allocate_bindings
        from trtutils.core._engine import create_engine
        from trtutils.core._stream import destroy_stream

        engine, context, _logger, stream = create_engine(simple_engine_path)
        try:
            inputs, outputs, _allocs = allocate_bindings(engine, context)

            for binding in inputs + outputs:
                assert binding.allocation != 0
                assert isinstance(binding.allocation, int)

            for b in inputs + outputs:
                b.free()
        finally:
            destroy_stream(stream)

    def test_bindings_have_host_allocation(self, simple_engine_path) -> None:
        """Each binding has a numpy array host allocation."""
        from trtutils.core._bindings import allocate_bindings
        from trtutils.core._engine import create_engine
        from trtutils.core._stream import destroy_stream

        engine, context, _logger, stream = create_engine(simple_engine_path)
        try:
            inputs, outputs, _allocs = allocate_bindings(engine, context)

            for binding in inputs + outputs:
                assert isinstance(binding.host_allocation, np.ndarray)

            for b in inputs + outputs:
                b.free()
        finally:
            destroy_stream(stream)

    def test_free_all_bindings(self, simple_engine_path) -> None:
        """All bindings can be freed without error."""
        from trtutils.core._bindings import allocate_bindings
        from trtutils.core._engine import create_engine
        from trtutils.core._stream import destroy_stream

        engine, context, _logger, stream = create_engine(simple_engine_path)
        try:
            inputs, outputs, _allocs = allocate_bindings(engine, context)

            for binding in inputs + outputs:
                binding.free()  # Should not raise
        finally:
            destroy_stream(stream)

    def test_bindings_input_output_flags(self, simple_engine_path) -> None:
        """Input bindings have is_input=True, outputs is_input=False."""
        from trtutils.core._bindings import allocate_bindings
        from trtutils.core._engine import create_engine
        from trtutils.core._stream import destroy_stream

        engine, context, _logger, stream = create_engine(simple_engine_path)
        try:
            inputs, outputs, _allocs = allocate_bindings(engine, context)

            for binding in inputs:
                assert binding.is_input is True
            for binding in outputs:
                assert binding.is_input is False

            for b in inputs + outputs:
                b.free()
        finally:
            destroy_stream(stream)


# ---------------------------------------------------------------------------
# Helper for FLAGS mocking
# ---------------------------------------------------------------------------
def _mock_flags(*, trt_10: bool):
    """Create a FLAGS mock with a specific TRT_10 value."""
    from trtutils._flags import FLAGS as REAL_FLAGS

    mock = MagicMock()
    for attr in dir(REAL_FLAGS):
        if not attr.startswith("_"):
            setattr(mock, attr, getattr(REAL_FLAGS, attr))
    mock.TRT_10 = trt_10
    mock.NVTX_ENABLED = False
    return mock


# ---------------------------------------------------------------------------
# allocate_bindings -- dynamic shape handling (TRT 10 + legacy paths)
# ---------------------------------------------------------------------------
@pytest.mark.gpu
class TestAllocateBindingsDynamicShapes:
    """Tests for allocate_bindings() dynamic batch dimension handling."""

    def test_trt10_dynamic_batch_resolves_max_profile(self) -> None:
        """TRT 10: dynamic batch resolves to max profile shape."""
        from trtutils.compat._libs import trt
        from trtutils.core._bindings import allocate_bindings

        mock_engine = MagicMock()
        mock_context = MagicMock()

        mock_engine.num_io_tensors = 2
        mock_engine.get_tensor_name.side_effect = ["input_0", "output_0"]
        mock_engine.get_tensor_mode.side_effect = [
            trt.TensorIOMode.INPUT,
            trt.TensorIOMode.OUTPUT,
        ]
        mock_engine.get_tensor_dtype.return_value = trt.DataType.FLOAT
        mock_engine.get_tensor_format.return_value = trt.TensorFormat.LINEAR
        mock_engine.num_optimization_profiles = 1

        max_shape = (8, 3, 8, 8)
        mock_engine.get_tensor_profile_shape.return_value = [
            (1, 3, 8, 8),
            (4, 3, 8, 8),
            max_shape,
        ]

        mock_context.get_tensor_shape.side_effect = [
            (-1, 3, 8, 8),  # input initial (dynamic)
            max_shape,  # input after set_input_shape
            (8, 3, 8, 8),  # output
        ]

        flags_mock = _mock_flags(trt_10=True)
        with patch("trtutils.core._bindings.FLAGS", flags_mock):
            inputs, outputs, allocations = allocate_bindings(mock_engine, mock_context)

        try:
            assert len(inputs) == 1
            assert len(outputs) == 1
            assert len(allocations) == 2
            assert inputs[0].shape == list(max_shape)
            mock_context.set_input_shape.assert_called_once_with("input_0", max_shape)
        finally:
            for b in inputs + outputs:
                b.free()

    def test_trt10_dynamic_batch_no_profiles_raises(self) -> None:
        """TRT 10: dynamic batch with no profiles raises RuntimeError."""
        from trtutils.compat._libs import trt
        from trtutils.core._bindings import allocate_bindings

        mock_engine = MagicMock()
        mock_context = MagicMock()

        mock_engine.num_io_tensors = 1
        mock_engine.get_tensor_name.return_value = "input_0"
        mock_engine.get_tensor_mode.return_value = trt.TensorIOMode.INPUT
        mock_engine.get_tensor_dtype.return_value = trt.DataType.FLOAT
        mock_engine.get_tensor_format.return_value = trt.TensorFormat.LINEAR
        mock_engine.num_optimization_profiles = 0

        mock_context.get_tensor_shape.return_value = (-1, 3, 8, 8)

        flags_mock = _mock_flags(trt_10=True)
        with patch("trtutils.core._bindings.FLAGS", flags_mock):
            with pytest.raises(RuntimeError, match="No optimization profiles found"):
                allocate_bindings(mock_engine, mock_context)

    def test_trt10_dynamic_batch_invalid_profile_length_raises(self) -> None:
        """TRT 10: profile shape with wrong length raises RuntimeError."""
        from trtutils.compat._libs import trt
        from trtutils.core._bindings import allocate_bindings

        mock_engine = MagicMock()
        mock_context = MagicMock()

        mock_engine.num_io_tensors = 1
        mock_engine.get_tensor_name.return_value = "input_0"
        mock_engine.get_tensor_mode.return_value = trt.TensorIOMode.INPUT
        mock_engine.get_tensor_dtype.return_value = trt.DataType.FLOAT
        mock_engine.get_tensor_format.return_value = trt.TensorFormat.LINEAR
        mock_engine.num_optimization_profiles = 1
        # Only 2 elements instead of required 3
        mock_engine.get_tensor_profile_shape.return_value = [
            (1, 3, 8, 8),
            (4, 3, 8, 8),
        ]

        mock_context.get_tensor_shape.return_value = (-1, 3, 8, 8)

        flags_mock = _mock_flags(trt_10=True)
        with patch("trtutils.core._bindings.FLAGS", flags_mock):
            with pytest.raises(RuntimeError, match="has 2 elements, expected 3"):
                allocate_bindings(mock_engine, mock_context)

    def test_legacy_dynamic_batch_resolves_max_profile(self) -> None:
        """Legacy (<TRT 10): dynamic batch resolves to max profile shape."""
        from trtutils.compat._libs import trt
        from trtutils.core._bindings import allocate_bindings

        mock_engine = MagicMock()
        mock_context = MagicMock()

        mock_engine.num_bindings = 2
        mock_engine.binding_is_input.side_effect = [True, False]
        mock_engine.get_binding_name.side_effect = ["input_0", "output_0"]
        mock_engine.get_binding_dtype.return_value = trt.DataType.FLOAT
        mock_engine.get_binding_format.return_value = trt.TensorFormat.LINEAR
        mock_engine.num_optimization_profiles = 1

        max_shape = (8, 3, 8, 8)
        mock_engine.get_profile_shape.return_value = [
            (1, 3, 8, 8),
            (4, 3, 8, 8),
            max_shape,
        ]

        mock_context.get_binding_shape.side_effect = [
            (-1, 3, 8, 8),  # input initial (dynamic)
            max_shape,  # input after set_binding_shape
            (8, 3, 8, 8),  # output
        ]

        flags_mock = _mock_flags(trt_10=False)
        with patch("trtutils.core._bindings.FLAGS", flags_mock):
            inputs, outputs, allocations = allocate_bindings(mock_engine, mock_context)

        try:
            assert len(inputs) == 1
            assert len(outputs) == 1
            assert len(allocations) == 2
            assert inputs[0].shape == list(max_shape)
            mock_context.set_binding_shape.assert_called_once_with(0, max_shape)
        finally:
            for b in inputs + outputs:
                b.free()

    def test_legacy_dynamic_batch_no_profiles_raises(self) -> None:
        """Legacy: dynamic batch with no profiles raises RuntimeError."""
        from trtutils.compat._libs import trt
        from trtutils.core._bindings import allocate_bindings

        mock_engine = MagicMock()
        mock_context = MagicMock()

        mock_engine.num_bindings = 1
        mock_engine.binding_is_input.return_value = True
        mock_engine.get_binding_name.return_value = "input_0"
        mock_engine.get_binding_dtype.return_value = trt.DataType.FLOAT
        mock_engine.get_binding_format.return_value = trt.TensorFormat.LINEAR
        mock_engine.num_optimization_profiles = 0

        mock_context.get_binding_shape.return_value = (-1, 3, 8, 8)

        flags_mock = _mock_flags(trt_10=False)
        with patch("trtutils.core._bindings.FLAGS", flags_mock):
            with pytest.raises(RuntimeError, match="No optimization profiles found"):
                allocate_bindings(mock_engine, mock_context)

    def test_legacy_dynamic_batch_invalid_profile_length_raises(self) -> None:
        """Legacy: profile shape with wrong length raises RuntimeError."""
        from trtutils.compat._libs import trt
        from trtutils.core._bindings import allocate_bindings

        mock_engine = MagicMock()
        mock_context = MagicMock()

        mock_engine.num_bindings = 1
        mock_engine.binding_is_input.return_value = True
        mock_engine.get_binding_name.return_value = "input_0"
        mock_engine.get_binding_dtype.return_value = trt.DataType.FLOAT
        mock_engine.get_binding_format.return_value = trt.TensorFormat.LINEAR
        mock_engine.num_optimization_profiles = 1
        # Only 1 element instead of required 3
        mock_engine.get_profile_shape.return_value = [
            (1, 3, 8, 8),
        ]

        mock_context.get_binding_shape.return_value = (-1, 3, 8, 8)

        flags_mock = _mock_flags(trt_10=False)
        with patch("trtutils.core._bindings.FLAGS", flags_mock):
            with pytest.raises(RuntimeError, match="has 1 elements, expected 3"):
                allocate_bindings(mock_engine, mock_context)


# ---------------------------------------------------------------------------
# allocate_bindings -- input/output validation error paths
# ---------------------------------------------------------------------------
@pytest.mark.gpu
class TestAllocateBindingsValidation:
    """Tests for allocate_bindings() empty input/output error paths."""

    def test_no_input_tensors_raises(self) -> None:
        """Engine with only outputs raises ValueError for missing inputs."""
        from trtutils.compat._libs import trt
        from trtutils.core._bindings import allocate_bindings

        mock_engine = MagicMock()
        mock_context = MagicMock()

        # Single tensor, classified as output
        mock_engine.num_io_tensors = 1
        mock_engine.get_tensor_name.return_value = "output_0"
        mock_engine.get_tensor_mode.return_value = trt.TensorIOMode.OUTPUT
        mock_engine.get_tensor_dtype.return_value = trt.DataType.FLOAT
        mock_engine.get_tensor_format.return_value = trt.TensorFormat.LINEAR
        mock_context.get_tensor_shape.return_value = (1, 3, 8, 8)

        flags_mock = _mock_flags(trt_10=True)
        mock_binding = MagicMock()
        with patch("trtutils.core._bindings.FLAGS", flags_mock), patch(
            "trtutils.core._bindings.create_binding", return_value=mock_binding
        ):
            with pytest.raises(ValueError, match="No input tensors found"):
                allocate_bindings(mock_engine, mock_context)

    def test_no_output_tensors_raises(self) -> None:
        """Engine with only inputs raises ValueError for missing outputs."""
        from trtutils.compat._libs import trt
        from trtutils.core._bindings import allocate_bindings

        mock_engine = MagicMock()
        mock_context = MagicMock()

        # Single tensor, classified as input
        mock_engine.num_io_tensors = 1
        mock_engine.get_tensor_name.return_value = "input_0"
        mock_engine.get_tensor_mode.return_value = trt.TensorIOMode.INPUT
        mock_engine.get_tensor_dtype.return_value = trt.DataType.FLOAT
        mock_engine.get_tensor_format.return_value = trt.TensorFormat.LINEAR
        mock_context.get_tensor_shape.return_value = (1, 3, 8, 8)

        flags_mock = _mock_flags(trt_10=True)
        mock_binding = MagicMock()
        with patch("trtutils.core._bindings.FLAGS", flags_mock), patch(
            "trtutils.core._bindings.create_binding", return_value=mock_binding
        ):
            with pytest.raises(ValueError, match="No output tensors found"):
                allocate_bindings(mock_engine, mock_context)

    def test_no_tensors_raises_no_inputs(self) -> None:
        """Engine with zero tensors raises ValueError for missing inputs."""
        from trtutils.core._bindings import allocate_bindings

        mock_engine = MagicMock()
        mock_context = MagicMock()
        mock_engine.num_io_tensors = 0

        flags_mock = _mock_flags(trt_10=True)
        with patch("trtutils.core._bindings.FLAGS", flags_mock):
            with pytest.raises(ValueError, match="No input tensors found"):
                allocate_bindings(mock_engine, mock_context)
