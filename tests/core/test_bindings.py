"""Tests for src/trtutils/core/_bindings.py -- Binding dataclass and allocation."""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Binding dataclass tests
# ---------------------------------------------------------------------------
@pytest.mark.gpu
class TestBindingDataclass:
    """Tests for the Binding dataclass fields and free() method."""

    def test_binding_fields(self):
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

    def test_binding_is_dataclass(self):
        """Binding is a dataclass instance."""
        import dataclasses

        from trtutils.core._bindings import create_binding

        arr = np.zeros((2, 2), dtype=np.float32)
        binding = create_binding(arr)
        assert dataclasses.is_dataclass(binding)
        binding.free()

    def test_binding_free(self):
        """free() deallocates memory without error."""
        from trtutils.core._bindings import create_binding

        arr = np.zeros((10,), dtype=np.float32)
        binding = create_binding(arr)
        binding.free()  # Should not raise

    def test_binding_free_idempotent(self):
        """Calling free() multiple times does not crash (via __del__)."""
        from trtutils.core._bindings import create_binding

        arr = np.zeros((10,), dtype=np.float32)
        binding = create_binding(arr)
        binding.free()
        # The __del__ will call free() again and suppress RuntimeError
        del binding  # Should not crash

    def test_binding_shape_matches(self):
        """Binding shape matches the input array shape."""
        from trtutils.core._bindings import create_binding

        shape = (1, 3, 64, 64)
        arr = np.zeros(shape, dtype=np.float32)
        binding = create_binding(arr)
        assert binding.shape == list(shape)
        binding.free()

    def test_binding_dtype_matches(self):
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
    def test_create_binding_dtypes(self, dtype):
        """create_binding works with various numpy dtypes."""
        from trtutils.core._bindings import create_binding

        arr = np.zeros((8,), dtype=dtype)
        binding = create_binding(arr)
        assert binding.dtype == dtype
        assert binding.allocation != 0
        assert isinstance(binding.host_allocation, np.ndarray)
        binding.free()

    def test_create_binding_default_pagelocked(self):
        """Defaults to pagelocked memory (pagelocked_mem=None -> True)."""
        from trtutils.core._bindings import create_binding

        arr = np.zeros((4,), dtype=np.float32)
        binding = create_binding(arr)
        assert binding.pagelocked_mem is True
        binding.free()

    def test_create_binding_no_pagelocked(self):
        """pagelocked_mem=False uses regular numpy arrays for host."""
        from trtutils.core._bindings import create_binding

        arr = np.zeros((4,), dtype=np.float32)
        binding = create_binding(arr, pagelocked_mem=False)
        assert binding.pagelocked_mem is False
        assert isinstance(binding.host_allocation, np.ndarray)
        binding.free()

    def test_create_binding_pagelocked_explicit(self):
        """pagelocked_mem=True uses page-locked memory."""
        from trtutils.core._bindings import create_binding

        arr = np.zeros((4,), dtype=np.float32)
        binding = create_binding(arr, pagelocked_mem=True)
        assert binding.pagelocked_mem is True
        binding.free()

    def test_create_binding_unified_memory(self):
        """pagelocked=True, unified=True uses the unified memory path."""
        from trtutils.core._bindings import create_binding

        arr = np.zeros((4,), dtype=np.float32)
        binding = create_binding(arr, pagelocked_mem=True, unified_mem=True)
        assert binding.unified_mem is True
        assert binding.allocation != 0
        binding.free()

    def test_create_binding_with_array_data(self):
        """use_array_data=True copies data from the input array."""
        from trtutils.core._bindings import create_binding

        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        binding = create_binding(arr, use_array_data=True)
        np.testing.assert_array_equal(binding.host_allocation, arr)
        binding.free()

    def test_create_binding_without_array_data(self):
        """use_array_data=None does not copy input data."""
        from trtutils.core._bindings import create_binding

        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        binding = create_binding(arr)
        # Host allocation should be zeros (data not copied)
        np.testing.assert_array_equal(binding.host_allocation, np.zeros_like(arr))
        binding.free()

    def test_create_binding_name_and_id(self):
        """create_binding stores the given name and bind_id."""
        from trtutils.core._bindings import create_binding

        arr = np.zeros((2,), dtype=np.float32)
        binding = create_binding(arr, bind_id=5, name="test_tensor")
        assert binding.index == 5
        assert binding.name == "test_tensor"
        binding.free()

    def test_create_binding_is_input(self):
        """create_binding stores the is_input flag."""
        from trtutils.core._bindings import create_binding

        arr = np.zeros((2,), dtype=np.float32)
        binding_input = create_binding(arr, is_input=True)
        binding_output = create_binding(arr, is_input=False)
        assert binding_input.is_input is True
        assert binding_output.is_input is False
        binding_input.free()
        binding_output.free()

    def test_create_binding_multidim_shape(self):
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
        "pagelocked,unified",
        [
            pytest.param(True, False, id="pagelocked"),
            pytest.param(False, False, id="default"),
            pytest.param(True, True, id="unified"),
        ],
    )
    def test_allocate_returns_io_bindings(self, simple_engine_path, pagelocked, unified):
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

    def test_bindings_count_matches_engine(self, simple_engine_path):
        """Number of input+output bindings matches engine tensor count."""
        from trtutils._flags import FLAGS
        from trtutils.core._bindings import allocate_bindings
        from trtutils.core._engine import create_engine
        from trtutils.core._stream import destroy_stream

        engine, context, _logger, stream = create_engine(simple_engine_path)
        try:
            inputs, outputs, allocations = allocate_bindings(engine, context)

            if FLAGS.TRT_10:
                expected = engine.num_io_tensors
            else:
                expected = engine.num_bindings

            assert len(inputs) + len(outputs) == expected
            assert len(allocations) == expected

            for b in inputs + outputs:
                b.free()
        finally:
            destroy_stream(stream)

    def test_bindings_have_device_allocation(self, simple_engine_path):
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

    def test_bindings_have_host_allocation(self, simple_engine_path):
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

    def test_free_all_bindings(self, simple_engine_path):
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

    def test_bindings_input_output_flags(self, simple_engine_path):
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
