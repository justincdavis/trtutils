# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/core/_bindings.py -- Binding dataclass and allocation."""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from trtutils._flags import FLAGS
from trtutils.compat._libs import trt
from trtutils.core import _bindings
from trtutils.core._bindings import allocate_bindings, create_binding


# ---------------------------------------------------------------------------
# Binding dataclass tests
# ---------------------------------------------------------------------------
class TestBindingDataclass:
    """Tests for the Binding dataclass fields and free() method."""

    def test_binding_fields(self) -> None:
        """Binding has all expected dataclass fields."""
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
        arr = np.zeros((2, 2), dtype=np.float32)
        binding = create_binding(arr)
        assert dataclasses.is_dataclass(binding)
        binding.free()

    def test_binding_free(self) -> None:
        """free() deallocates memory without error."""
        arr = np.zeros((10,), dtype=np.float32)
        binding = create_binding(arr)
        binding.free()  # Should not raise

    def test_binding_free_idempotent(self) -> None:
        """Calling free() multiple times does not crash (via __del__)."""
        arr = np.zeros((10,), dtype=np.float32)
        binding = create_binding(arr)
        binding.free()
        # The __del__ will call free() again and suppress RuntimeError
        del binding  # Should not crash

    def test_binding_shape_matches(self) -> None:
        """Binding shape matches the input array shape."""
        shape = (1, 3, 64, 64)
        arr = np.zeros(shape, dtype=np.float32)
        binding = create_binding(arr)
        assert binding.shape == list(shape)
        binding.free()

    def test_binding_dtype_matches(self) -> None:
        """Binding dtype matches the input array dtype."""
        arr = np.zeros((4, 4), dtype=np.float16)
        binding = create_binding(arr)
        assert binding.dtype == np.float16
        binding.free()


# ---------------------------------------------------------------------------
# create_binding tests
# ---------------------------------------------------------------------------
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
        arr = np.zeros((8,), dtype=dtype)
        binding = create_binding(arr)
        assert binding.dtype == dtype
        assert binding.allocation != 0
        assert isinstance(binding.host_allocation, np.ndarray)
        binding.free()

    @pytest.mark.parametrize(
        ("pagelocked_arg", "expected"),
        [
            pytest.param(None, True, id="default"),
            pytest.param(False, False, id="disabled"),
            pytest.param(True, True, id="enabled"),
        ],
    )
    def test_create_binding_pagelocked(self, pagelocked_arg, expected) -> None:
        """create_binding pagelocked_mem parameter variants."""
        arr = np.zeros((4,), dtype=np.float32)
        kwargs = {} if pagelocked_arg is None else {"pagelocked_mem": pagelocked_arg}
        binding = create_binding(arr, **kwargs)
        assert binding.pagelocked_mem is expected
        binding.free()

    def test_create_binding_unified_memory(self) -> None:
        """pagelocked=True, unified=True uses the unified memory path."""
        arr = np.zeros((4,), dtype=np.float32)
        binding = create_binding(arr, pagelocked_mem=True, unified_mem=True)
        assert binding.unified_mem is True
        assert binding.allocation != 0
        binding.free()

    def test_create_binding_unified_memory_frees_host_only(self) -> None:
        """Mapped unified host allocations should not be freed with cudaFree."""
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

    @pytest.mark.parametrize(
        ("use_array_data", "expect_data"),
        [
            pytest.param(True, True, id="with_data"),
            pytest.param(None, False, id="without_data"),
        ],
    )
    def test_create_binding_array_data(self, use_array_data, expect_data) -> None:
        """create_binding use_array_data parameter controls data copy."""
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        kwargs = {"use_array_data": use_array_data} if use_array_data is not None else {}
        binding = create_binding(arr, **kwargs)
        if expect_data:
            np.testing.assert_array_equal(binding.host_allocation, arr)
        else:
            np.testing.assert_array_equal(binding.host_allocation, np.zeros_like(arr))
        binding.free()

    def test_create_binding_name_and_id(self) -> None:
        """create_binding stores the given name and bind_id."""
        arr = np.zeros((2,), dtype=np.float32)
        binding = create_binding(arr, bind_id=5, name="test_tensor")
        assert binding.index == 5
        assert binding.name == "test_tensor"
        binding.free()

    def test_create_binding_is_input(self) -> None:
        """create_binding stores the is_input flag."""
        arr = np.zeros((2,), dtype=np.float32)
        binding_input = create_binding(arr, is_input=True)
        binding_output = create_binding(arr, is_input=False)
        assert binding_input.is_input is True
        assert binding_output.is_input is False
        binding_input.free()
        binding_output.free()

    def test_create_binding_multidim_shape(self) -> None:
        """create_binding preserves multi-dimensional shapes."""
        arr = np.zeros((1, 3, 224, 224), dtype=np.float32)
        binding = create_binding(arr)
        assert binding.shape == [1, 3, 224, 224]
        assert binding.host_allocation.shape == (1, 3, 224, 224)
        binding.free()


# ---------------------------------------------------------------------------
# allocate_bindings tests (requires a real TensorRT engine)
# ---------------------------------------------------------------------------
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
    def test_allocate_returns_io_bindings(self, simple_engine, pagelocked, unified: object) -> None:
        """allocate_bindings returns (inputs, outputs, allocations)."""
        engine, context, _stream = simple_engine
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

    def test_bindings_count_matches_engine(self, simple_engine) -> None:
        """Number of input+output bindings matches engine tensor count."""
        engine, context, _stream = simple_engine
        inputs, outputs, allocations = allocate_bindings(engine, context)

        expected = engine.num_io_tensors if FLAGS.TRT_10 else engine.num_bindings

        assert len(inputs) + len(outputs) == expected
        assert len(allocations) == expected

        for b in inputs + outputs:
            b.free()

    def test_bindings_have_device_allocation(self, simple_engine) -> None:
        """Each binding has a non-zero device allocation pointer."""
        engine, context, _stream = simple_engine
        inputs, outputs, _allocs = allocate_bindings(engine, context)

        for binding in inputs + outputs:
            assert binding.allocation != 0
            assert isinstance(binding.allocation, int)

        for b in inputs + outputs:
            b.free()

    def test_bindings_have_host_allocation(self, simple_engine) -> None:
        """Each binding has a numpy array host allocation."""
        engine, context, _stream = simple_engine
        inputs, outputs, _allocs = allocate_bindings(engine, context)

        for binding in inputs + outputs:
            assert isinstance(binding.host_allocation, np.ndarray)

        for b in inputs + outputs:
            b.free()

    def test_free_all_bindings(self, simple_engine) -> None:
        """All bindings can be freed without error."""
        engine, context, _stream = simple_engine
        inputs, outputs, _allocs = allocate_bindings(engine, context)

        for binding in inputs + outputs:
            binding.free()  # Should not raise

    def test_bindings_input_output_flags(self, simple_engine) -> None:
        """Input bindings have is_input=True, outputs is_input=False."""
        engine, context, _stream = simple_engine
        inputs, outputs, _allocs = allocate_bindings(engine, context)

        for binding in inputs:
            assert binding.is_input is True
        for binding in outputs:
            assert binding.is_input is False

        for b in inputs + outputs:
            b.free()


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
class TestAllocateBindingsDynamicShapes:
    """Tests for allocate_bindings() dynamic batch dimension handling."""

    def test_trt10_dynamic_batch_resolves_max_profile(self) -> None:
        """TRT 10: dynamic batch resolves to max profile shape."""
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
class TestAllocateBindingsValidation:
    """Tests for allocate_bindings() empty input/output error paths."""

    @pytest.mark.parametrize(
        ("tensor_mode", "match_msg"),
        [
            pytest.param(trt.TensorIOMode.OUTPUT, "No input tensors found", id="no_inputs"),
            pytest.param(trt.TensorIOMode.INPUT, "No output tensors found", id="no_outputs"),
        ],
    )
    def test_missing_io_tensors_raises(self, tensor_mode, match_msg) -> None:
        """Engine missing inputs or outputs raises ValueError."""
        mock_engine = MagicMock()
        mock_context = MagicMock()
        mock_engine.num_io_tensors = 1
        mock_engine.get_tensor_name.return_value = "tensor_0"
        mock_engine.get_tensor_mode.return_value = tensor_mode
        mock_engine.get_tensor_dtype.return_value = trt.DataType.FLOAT
        mock_engine.get_tensor_format.return_value = trt.TensorFormat.LINEAR
        mock_context.get_tensor_shape.return_value = (1, 3, 8, 8)
        flags_mock = _mock_flags(trt_10=True)
        mock_binding = MagicMock()
        with patch("trtutils.core._bindings.FLAGS", flags_mock), patch(
            "trtutils.core._bindings.create_binding", return_value=mock_binding
        ):
            with pytest.raises(ValueError, match=match_msg):
                allocate_bindings(mock_engine, mock_context)

    def test_no_tensors_raises_no_inputs(self) -> None:
        """Engine with zero tensors raises ValueError for missing inputs."""
        mock_engine = MagicMock()
        mock_context = MagicMock()
        mock_engine.num_io_tensors = 0

        flags_mock = _mock_flags(trt_10=True)
        with patch("trtutils.core._bindings.FLAGS", flags_mock):
            with pytest.raises(ValueError, match="No input tensors found"):
                allocate_bindings(mock_engine, mock_context)
