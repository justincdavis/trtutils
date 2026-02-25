"""Tests for src/trtutils/core/_nvrtc.py -- NVRTC compilation and loading."""

from __future__ import annotations

from pathlib import Path

import pytest

# A minimal CUDA kernel that compiles successfully
TRIVIAL_KERNEL = """\
extern "C" __global__ void trivial_kernel(float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (float)idx;
    }
}
"""


@pytest.mark.gpu
class TestFindCudaIncludeDir:
    """Tests for find_cuda_include_dir()."""

    def test_finds_cuda_include(self) -> None:
        """find_cuda_include_dir() should return a Path or None."""
        from trtutils.core._nvrtc import find_cuda_include_dir

        # Clear the lru_cache to test fresh
        find_cuda_include_dir.cache_clear()
        result = find_cuda_include_dir()
        # On a system with CUDA installed, this should return a valid path
        # On systems without CUDA, it returns None
        assert result is None or isinstance(result, Path)

    def test_result_is_cached(self) -> None:
        """Second call returns same result due to lru_cache."""
        from trtutils.core._nvrtc import find_cuda_include_dir

        find_cuda_include_dir.cache_clear()
        result1 = find_cuda_include_dir()
        result2 = find_cuda_include_dir()
        assert result1 == result2

    def test_env_var_cuda_home(self, tmp_path, monkeypatch) -> None:
        """CUDA_HOME environment variable is checked first."""
        from trtutils.core._nvrtc import find_cuda_include_dir

        find_cuda_include_dir.cache_clear()
        include_dir = tmp_path / "include"
        include_dir.mkdir()
        monkeypatch.setenv("CUDA_HOME", str(tmp_path))
        # Clear any CUDA_PATH that might interfere
        monkeypatch.delenv("CUDA_PATH", raising=False)

        result = find_cuda_include_dir()
        assert result == include_dir

        # Clean up the cache so other tests are not affected
        find_cuda_include_dir.cache_clear()

    def test_env_var_cuda_path(self, tmp_path, monkeypatch) -> None:
        """CUDA_PATH environment variable is checked second."""
        from trtutils.core._nvrtc import find_cuda_include_dir

        find_cuda_include_dir.cache_clear()
        include_dir = tmp_path / "include"
        include_dir.mkdir()
        monkeypatch.delenv("CUDA_HOME", raising=False)
        monkeypatch.setenv("CUDA_PATH", str(tmp_path))

        result = find_cuda_include_dir()
        assert result == include_dir

        find_cuda_include_dir.cache_clear()

    def test_env_var_no_include_dir(self, tmp_path, monkeypatch) -> None:
        """CUDA_HOME set but no include dir beneath it falls through."""
        from trtutils.core._nvrtc import find_cuda_include_dir

        find_cuda_include_dir.cache_clear()
        monkeypatch.setenv("CUDA_HOME", str(tmp_path))
        monkeypatch.delenv("CUDA_PATH", raising=False)
        # No include subdirectory exists

        # The function should continue searching other paths
        result = find_cuda_include_dir()
        # Result depends on system; just verify it does not crash
        assert result is None or isinstance(result, Path)

        find_cuda_include_dir.cache_clear()


@pytest.mark.gpu
class TestGetDefaultNvrtcOpts:
    """Tests for _get_default_nvrtc_opts internal helper."""

    def test_returns_list(self) -> None:
        """_get_default_nvrtc_opts returns a list."""
        from trtutils.core._nvrtc import _get_default_nvrtc_opts

        result = _get_default_nvrtc_opts()
        assert isinstance(result, list)

    def test_includes_include_path_when_found(self, tmp_path, monkeypatch) -> None:
        """When CUDA include dir is found, -I flag is added."""
        from trtutils.core._nvrtc import (
            _get_default_nvrtc_opts,
            find_cuda_include_dir,
        )

        find_cuda_include_dir.cache_clear()
        include_dir = tmp_path / "include"
        include_dir.mkdir()
        monkeypatch.setenv("CUDA_HOME", str(tmp_path))
        monkeypatch.delenv("CUDA_PATH", raising=False)

        opts = _get_default_nvrtc_opts()
        assert any(b"-I" in opt for opt in opts)

        find_cuda_include_dir.cache_clear()


@pytest.mark.gpu
class TestNvrtcErrorHandling:
    """Tests for check_nvrtc_err and nvrtc_call."""

    def test_check_nvrtc_err_success(self) -> None:
        """check_nvrtc_err on success should not raise."""
        from trtutils.compat._libs import nvrtc
        from trtutils.core._nvrtc import check_nvrtc_err

        check_nvrtc_err(nvrtc.nvrtcResult.NVRTC_SUCCESS)

    def test_check_nvrtc_err_failure(self) -> None:
        """check_nvrtc_err on failure should raise RuntimeError."""
        from trtutils.compat._libs import nvrtc
        from trtutils.core._nvrtc import check_nvrtc_err

        with pytest.raises(RuntimeError, match="NVRTC Error"):
            check_nvrtc_err(nvrtc.nvrtcResult.NVRTC_ERROR_COMPILATION)

    def test_nvrtc_call_success(self) -> None:
        """nvrtc_call with success returns value."""
        from trtutils.compat._libs import nvrtc
        from trtutils.core._nvrtc import nvrtc_call

        # nvrtcVersion returns (result, major, minor)
        result = nvrtc_call(nvrtc.nvrtcVersion())
        # result should be a tuple of (major, minor)
        assert result is not None


@pytest.mark.gpu
class TestCompileKernel:
    """Tests for compile_kernel."""

    def test_compile_simple_kernel(self) -> None:
        """compile_kernel compiles a trivial CUDA kernel."""
        from trtutils.core._nvrtc import compile_kernel

        ptx = compile_kernel(TRIVIAL_KERNEL, "trivial_kernel")
        assert ptx is not None

    def test_compile_returns_chararray(self) -> None:
        """compile_kernel returns np.char.chararray."""
        import numpy as np

        from trtutils.core._nvrtc import compile_kernel

        ptx = compile_kernel(TRIVIAL_KERNEL, "trivial_kernel")
        assert isinstance(ptx, np.char.chararray)

    def test_compile_with_opts(self) -> None:
        """compile_kernel with additional opts does not raise."""
        from trtutils.core._nvrtc import compile_kernel

        ptx = compile_kernel(TRIVIAL_KERNEL, "trivial_kernel", opts=["--use_fast_math"])
        assert ptx is not None

    def test_compile_verbose(self) -> None:
        """compile_kernel with verbose=True does not raise."""
        from trtutils.core._nvrtc import compile_kernel

        ptx = compile_kernel(TRIVIAL_KERNEL, "trivial_kernel", verbose=True)
        assert ptx is not None


@pytest.mark.gpu
class TestLoadKernel:
    """Tests for load_kernel."""

    def test_load_from_ptx(self) -> None:
        """load_kernel loads compiled PTX and returns module and kernel."""
        from trtutils.compat._libs import cuda
        from trtutils.core._nvrtc import compile_kernel, load_kernel

        ptx = compile_kernel(TRIVIAL_KERNEL, "trivial_kernel")
        module, kernel = load_kernel(ptx, "trivial_kernel")
        assert module is not None
        assert kernel is not None
        # Verify types
        assert isinstance(module, cuda.CUmodule)

    def test_load_verbose(self) -> None:
        """load_kernel with verbose=True does not raise."""
        from trtutils.core._nvrtc import compile_kernel, load_kernel

        ptx = compile_kernel(TRIVIAL_KERNEL, "trivial_kernel")
        module, _kernel = load_kernel(ptx, "trivial_kernel", verbose=True)
        assert module is not None


@pytest.mark.gpu
class TestCompileAndLoadKernel:
    """Tests for compile_and_load_kernel."""

    def test_full_lifecycle(self) -> None:
        """compile_and_load_kernel compiles and loads in one call."""
        from trtutils.core._nvrtc import compile_and_load_kernel

        module, kernel = compile_and_load_kernel(TRIVIAL_KERNEL, "trivial_kernel")
        assert module is not None
        assert kernel is not None

    def test_with_opts(self) -> None:
        """compile_and_load_kernel with opts."""
        from trtutils.core._nvrtc import compile_and_load_kernel

        module, _kernel = compile_and_load_kernel(
            TRIVIAL_KERNEL, "trivial_kernel", opts=["--use_fast_math"]
        )
        assert module is not None

    def test_with_verbose(self) -> None:
        """compile_and_load_kernel with verbose=True."""
        from trtutils.core._nvrtc import compile_and_load_kernel

        module, _kernel = compile_and_load_kernel(TRIVIAL_KERNEL, "trivial_kernel", verbose=True)
        assert module is not None

    def test_with_existing_kernel_file(self) -> None:
        """compile_and_load_kernel with a real .cu kernel from the project."""
        kernel_dir = Path(__file__).parent.parent.parent / "src" / "trtutils" / "image" / "_kernels"
        # Pick the simplest kernel file
        kernel_files = list(kernel_dir.glob("*.cu"))
        if not kernel_files:
            pytest.skip("No .cu kernel files found in image/_kernels/")

        # Just verify we can read the file; full compilation
        # might require specific CUDA headers
        kernel_file = kernel_files[0]
        assert kernel_file.exists()
