"""Tests for src/trtutils/core/_kernels.py -- kernel compilation and args."""

from __future__ import annotations

import numpy as np
import pytest

# A minimal CUDA kernel for testing the Kernel class and launch_kernel
TRIVIAL_KERNEL_CODE = """\
extern "C" __global__ void trivial_kernel(float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (float)idx;
    }
}
"""


# ---------------------------------------------------------------------------
# create_kernel_args tests (does not require GPU launch, but needs numpy)
# ---------------------------------------------------------------------------
@pytest.mark.gpu
class TestCreateKernelArgs:
    """Tests for create_kernel_args() argument conversion."""

    def test_int_arg(self) -> None:
        """Integer arguments are converted to uint64 arrays."""
        from trtutils.core._kernels import create_kernel_args

        ptrs, intermediates = create_kernel_args(42)
        assert isinstance(ptrs, np.ndarray)
        assert ptrs.dtype == np.uint64
        assert len(ptrs) == 1
        assert len(intermediates) == 1
        assert intermediates[0].dtype == np.uint64

    def test_float_arg(self) -> None:
        """Float arguments are converted to float32 arrays."""
        from trtutils.core._kernels import create_kernel_args

        ptrs, intermediates = create_kernel_args(3.14)
        assert isinstance(ptrs, np.ndarray)
        assert len(ptrs) == 1
        assert intermediates[0].dtype == np.float32

    def test_ndarray_arg(self) -> None:
        """Pre-formed np.ndarray arguments are passed through."""
        from trtutils.core._kernels import create_kernel_args

        arr = np.array([100], dtype=np.uint64)
        ptrs, intermediates = create_kernel_args(arr)
        assert len(ptrs) == 1
        assert intermediates[0] is arr

    def test_multiple_args(self) -> None:
        """Multiple arguments produce correct number of pointers."""
        from trtutils.core._kernels import create_kernel_args

        ptrs, intermediates = create_kernel_args(10, 20.0, 30)
        assert len(ptrs) == 3
        assert len(intermediates) == 3

    def test_unsupported_type_raises(self) -> None:
        """Passing an unsupported type raises TypeError."""
        from trtutils.core._kernels import create_kernel_args

        with pytest.raises(TypeError, match="Unrecognized arg type"):
            create_kernel_args("not_a_valid_arg")

    def test_verbose_output(self) -> None:
        """verbose=True does not raise (exercises LOG.debug paths)."""
        from trtutils.core._kernels import create_kernel_args

        ptrs, _intermediates = create_kernel_args(42, 3.14, verbose=True)
        assert len(ptrs) == 2

    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(np.float32, id="float32"),
            pytest.param(np.float16, id="float16"),
            pytest.param(np.int32, id="int32"),
            pytest.param(np.uint8, id="uint8"),
        ],
    )
    def test_ndarray_various_dtypes(self, dtype) -> None:
        """np.ndarray args with various dtypes are passed through."""
        from trtutils.core._kernels import create_kernel_args

        arr = np.array([1], dtype=dtype)
        ptrs, intermediates = create_kernel_args(arr)
        assert len(ptrs) == 1
        assert intermediates[0] is arr


# ---------------------------------------------------------------------------
# Kernel class tests (requires GPU for compilation and launch)
# ---------------------------------------------------------------------------
@pytest.mark.gpu
class TestKernelCompilation:
    """Tests for Kernel class compilation from .cu source."""

    def test_compile_from_cu_file(self, tmp_path) -> None:
        """Kernel compiles from a .cu file path."""
        from trtutils.core._kernels import Kernel

        cu_file = tmp_path / "trivial_kernel.cu"
        cu_file.write_text(TRIVIAL_KERNEL_CODE)

        kernel = Kernel(cu_file, "trivial_kernel")
        assert kernel._kernel is not None
        assert kernel._module is not None
        kernel.free()

    def test_compile_from_string_path(self, tmp_path) -> None:
        """Kernel accepts a string path (not just Path objects)."""
        from trtutils.core._kernels import Kernel

        cu_file = tmp_path / "trivial_kernel.cu"
        cu_file.write_text(TRIVIAL_KERNEL_CODE)

        kernel = Kernel(str(cu_file), "trivial_kernel")
        assert kernel._kernel is not None
        kernel.free()

    def test_kernel_has_function(self, tmp_path) -> None:
        """Kernel._kernel attribute is set after compilation."""
        from trtutils.core._kernels import Kernel

        cu_file = tmp_path / "trivial_kernel.cu"
        cu_file.write_text(TRIVIAL_KERNEL_CODE)

        kernel = Kernel(cu_file, "trivial_kernel")
        assert kernel._kernel is not None
        kernel.free()

    def test_kernel_has_module(self, tmp_path) -> None:
        """Kernel._module attribute is set after compilation."""
        from trtutils.core._kernels import Kernel

        cu_file = tmp_path / "trivial_kernel.cu"
        cu_file.write_text(TRIVIAL_KERNEL_CODE)

        kernel = Kernel(cu_file, "trivial_kernel")
        assert kernel._module is not None
        kernel.free()

    def test_free_unloads(self, tmp_path) -> None:
        """free() sets _module to None and _freed to True."""
        from trtutils.core._kernels import Kernel

        cu_file = tmp_path / "trivial_kernel.cu"
        cu_file.write_text(TRIVIAL_KERNEL_CODE)

        kernel = Kernel(cu_file, "trivial_kernel")
        kernel.free()
        assert kernel._module is None
        assert kernel._freed is True

    def test_free_idempotent(self, tmp_path) -> None:
        """Calling free() twice does not raise."""
        from trtutils.core._kernels import Kernel

        cu_file = tmp_path / "trivial_kernel.cu"
        cu_file.write_text(TRIVIAL_KERNEL_CODE)

        kernel = Kernel(cu_file, "trivial_kernel")
        kernel.free()
        kernel.free()  # Should not raise

    def test_del_calls_free(self, tmp_path) -> None:
        """__del__ cleans up without error."""
        from trtutils.core._kernels import Kernel

        cu_file = tmp_path / "trivial_kernel.cu"
        cu_file.write_text(TRIVIAL_KERNEL_CODE)

        kernel = Kernel(cu_file, "trivial_kernel")
        del kernel  # Should not crash

    def test_compile_with_verbose(self, tmp_path) -> None:
        """Kernel with verbose=True does not raise."""
        from trtutils.core._kernels import Kernel

        cu_file = tmp_path / "trivial_kernel.cu"
        cu_file.write_text(TRIVIAL_KERNEL_CODE)

        kernel = Kernel(cu_file, "trivial_kernel", verbose=True)
        assert kernel._kernel is not None
        kernel.free()


# ---------------------------------------------------------------------------
# Kernel.create_args and arg caching tests
# ---------------------------------------------------------------------------
@pytest.mark.gpu
class TestKernelArgs:
    """Tests for Kernel.create_args() and the deque-based arg cache."""

    def test_create_args_returns_ndarray(self, tmp_path) -> None:
        """create_args returns a numpy array of pointers."""
        from trtutils.core._kernels import Kernel

        cu_file = tmp_path / "trivial_kernel.cu"
        cu_file.write_text(TRIVIAL_KERNEL_CODE)

        kernel = Kernel(cu_file, "trivial_kernel")
        args = kernel.create_args(42, 10)
        assert isinstance(args, np.ndarray)
        assert args.dtype == np.uint64
        assert len(args) == 2
        kernel.free()

    def test_args_caching(self, tmp_path) -> None:
        """The deque cache retains intermediate arg arrays."""
        from trtutils.core._kernels import Kernel

        cu_file = tmp_path / "trivial_kernel.cu"
        cu_file.write_text(TRIVIAL_KERNEL_CODE)

        kernel = Kernel(cu_file, "trivial_kernel", max_arg_cache=2)
        kernel.create_args(1, 2)
        assert len(kernel._inter_args) == 1
        kernel.create_args(3, 4)
        assert len(kernel._inter_args) == 2
        # Third call should evict the oldest
        kernel.create_args(5, 6)
        assert len(kernel._inter_args) == 2
        kernel.free()

    def test_create_args_pointer_arg(self, tmp_path) -> None:
        """Integer pointer args are handled correctly."""
        from trtutils.core._kernels import Kernel
        from trtutils.core._memory import cuda_free, cuda_malloc

        cu_file = tmp_path / "trivial_kernel.cu"
        cu_file.write_text(TRIVIAL_KERNEL_CODE)

        kernel = Kernel(cu_file, "trivial_kernel")
        ptr = cuda_malloc(1024)
        args = kernel.create_args(ptr, 256)
        assert len(args) == 2
        kernel.free()
        cuda_free(ptr)


# ---------------------------------------------------------------------------
# Kernel launch tests
# ---------------------------------------------------------------------------
@pytest.mark.gpu
class TestKernelLaunch:
    """Tests for launching a compiled kernel."""

    def test_launch_simple_kernel(self, tmp_path, cuda_stream) -> None:
        """Execute a compiled kernel via launch_kernel."""
        from trtutils.core._kernels import Kernel, launch_kernel
        from trtutils.core._memory import (
            cuda_free,
            cuda_malloc,
            memcpy_device_to_host,
        )
        from trtutils.core._stream import stream_synchronize

        cu_file = tmp_path / "trivial_kernel.cu"
        cu_file.write_text(TRIVIAL_KERNEL_CODE)

        kernel = Kernel(cu_file, "trivial_kernel")

        n = 32
        nbytes = n * np.dtype(np.float32).itemsize
        d_out = cuda_malloc(nbytes)

        args = kernel.create_args(d_out, n)
        launch_kernel(
            kernel._kernel,
            (1, 1, 1),
            (n, 1, 1),
            cuda_stream,
            args,
        )
        stream_synchronize(cuda_stream)

        result = np.zeros(n, dtype=np.float32)
        memcpy_device_to_host(result, d_out)

        expected = np.arange(n, dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

        kernel.free()
        cuda_free(d_out)

    def test_launch_via_call(self, tmp_path, cuda_stream) -> None:
        """Execute a kernel via Kernel.__call__ (the call method)."""
        from trtutils.core._kernels import Kernel
        from trtutils.core._memory import (
            cuda_free,
            cuda_malloc,
            memcpy_device_to_host,
        )
        from trtutils.core._stream import stream_synchronize

        cu_file = tmp_path / "trivial_kernel.cu"
        cu_file.write_text(TRIVIAL_KERNEL_CODE)

        kernel = Kernel(cu_file, "trivial_kernel")

        n = 16
        nbytes = n * np.dtype(np.float32).itemsize
        d_out = cuda_malloc(nbytes)

        args = kernel.create_args(d_out, n)
        kernel((1, 1, 1), (n, 1, 1), cuda_stream, args)
        stream_synchronize(cuda_stream)

        result = np.zeros(n, dtype=np.float32)
        memcpy_device_to_host(result, d_out)

        expected = np.arange(n, dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

        kernel.free()
        cuda_free(d_out)

    def test_launch_with_verbose(self, tmp_path, cuda_stream) -> None:
        """Kernel.call with verbose=True does not raise."""
        from trtutils.core._kernels import Kernel
        from trtutils.core._memory import cuda_free, cuda_malloc
        from trtutils.core._stream import stream_synchronize

        cu_file = tmp_path / "trivial_kernel.cu"
        cu_file.write_text(TRIVIAL_KERNEL_CODE)

        kernel = Kernel(cu_file, "trivial_kernel")

        n = 8
        d_out = cuda_malloc(n * np.dtype(np.float32).itemsize)
        args = kernel.create_args(d_out, n)
        kernel.call((1, 1, 1), (n, 1, 1), cuda_stream, args, verbose=True)
        stream_synchronize(cuda_stream)

        kernel.free()
        cuda_free(d_out)
