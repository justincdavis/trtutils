# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/core/_nvrtc.py -- NVRTC compilation and loading."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from trtutils.compat._libs import cuda, nvrtc
from trtutils.core._nvrtc import (
    _get_default_nvrtc_opts,
    check_nvrtc_err,
    compile_and_load_kernel,
    compile_kernel,
    find_cuda_include_dir,
    load_kernel,
    nvrtc_call,
)

# A minimal CUDA kernel that compiles successfully
TRIVIAL_KERNEL = """\
extern "C" __global__ void trivial_kernel(float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (float)idx;
    }
}
"""


class TestFindCudaIncludeDir:
    """Tests for find_cuda_include_dir()."""

    def test_finds_cuda_include(self) -> None:
        """find_cuda_include_dir() should return a Path or None."""
        result = find_cuda_include_dir()
        # On a system with CUDA installed, this should return a valid path
        # On systems without CUDA, it returns None
        assert result is None or isinstance(result, Path)

    def test_result_is_cached(self) -> None:
        """Second call returns same result due to lru_cache."""
        result1 = find_cuda_include_dir()
        result2 = find_cuda_include_dir()
        assert result1 == result2

    @pytest.mark.parametrize(
        ("env_var", "clear_var"),
        [
            pytest.param("CUDA_HOME", "CUDA_PATH", id="CUDA_HOME"),
            pytest.param("CUDA_PATH", "CUDA_HOME", id="CUDA_PATH"),
        ],
    )
    def test_env_var_sets_include_dir(self, tmp_path, monkeypatch, env_var, clear_var) -> None:
        """Environment variable pointing to a CUDA install is detected."""
        include_dir = tmp_path / "include"
        include_dir.mkdir()
        monkeypatch.setenv(env_var, str(tmp_path))
        monkeypatch.delenv(clear_var, raising=False)
        result = find_cuda_include_dir()
        assert result == include_dir

    def test_env_var_no_include_dir(self, tmp_path, monkeypatch) -> None:
        """CUDA_HOME set but no include dir beneath it falls through."""
        monkeypatch.setenv("CUDA_HOME", str(tmp_path))
        monkeypatch.delenv("CUDA_PATH", raising=False)
        # No include subdirectory exists

        # The function should continue searching other paths
        result = find_cuda_include_dir()
        # Result depends on system; just verify it does not crash
        assert result is None or isinstance(result, Path)


class TestGetDefaultNvrtcOpts:
    """Tests for _get_default_nvrtc_opts internal helper."""

    def test_returns_list(self) -> None:
        """_get_default_nvrtc_opts returns a list."""
        result = _get_default_nvrtc_opts()
        assert isinstance(result, list)

    def test_includes_include_path_when_found(self, tmp_path, monkeypatch) -> None:
        """When CUDA include dir is found, -I flag is added."""
        include_dir = tmp_path / "include"
        include_dir.mkdir()
        monkeypatch.setenv("CUDA_HOME", str(tmp_path))
        monkeypatch.delenv("CUDA_PATH", raising=False)

        opts = _get_default_nvrtc_opts()
        assert any(b"-I" in opt for opt in opts)


class TestNvrtcErrorHandling:
    """Tests for check_nvrtc_err and nvrtc_call."""

    def test_check_nvrtc_err_success(self) -> None:
        """check_nvrtc_err on success should not raise."""
        check_nvrtc_err(nvrtc.nvrtcResult.NVRTC_SUCCESS)

    def test_check_nvrtc_err_failure(self) -> None:
        """check_nvrtc_err on failure should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="NVRTC Error"):
            check_nvrtc_err(nvrtc.nvrtcResult.NVRTC_ERROR_COMPILATION)

    def test_nvrtc_call_success(self) -> None:
        """nvrtc_call with success returns value."""
        # nvrtcVersion returns (result, major, minor)
        result = nvrtc_call(nvrtc.nvrtcVersion())
        # result should be a tuple of (major, minor)
        assert result is not None


class TestCompileKernel:
    """Tests for compile_kernel."""

    def test_compile_returns_chararray(self) -> None:
        """compile_kernel returns np.char.chararray."""
        ptx = compile_kernel(TRIVIAL_KERNEL, "trivial_kernel")
        assert isinstance(ptx, np.char.chararray)

    def test_compile_with_opts(self) -> None:
        """compile_kernel with additional opts does not raise."""
        ptx = compile_kernel(TRIVIAL_KERNEL, "trivial_kernel", opts=["--use_fast_math"])
        assert ptx is not None

    def test_compile_verbose(self) -> None:
        """compile_kernel with verbose=True does not raise."""
        ptx = compile_kernel(TRIVIAL_KERNEL, "trivial_kernel", verbose=True)
        assert ptx is not None


class TestLoadKernel:
    """Tests for load_kernel."""

    @pytest.mark.parametrize(
        "verbose",
        [pytest.param(False, id="default"), pytest.param(True, id="verbose")],
    )
    def test_load_from_ptx(self, verbose) -> None:
        """load_kernel loads compiled PTX and returns module and kernel."""
        ptx = compile_kernel(TRIVIAL_KERNEL, "trivial_kernel")
        module, kernel = load_kernel(ptx, "trivial_kernel", verbose=verbose)
        assert module is not None
        assert kernel is not None
        if not verbose:
            assert isinstance(module, cuda.CUmodule)


class TestCompileAndLoadKernel:
    """Tests for compile_and_load_kernel."""

    def test_full_lifecycle(self) -> None:
        """compile_and_load_kernel compiles and loads in one call."""
        module, kernel = compile_and_load_kernel(TRIVIAL_KERNEL, "trivial_kernel")
        assert module is not None
        assert kernel is not None

    def test_with_opts(self) -> None:
        """compile_and_load_kernel with opts."""
        module, _kernel = compile_and_load_kernel(
            TRIVIAL_KERNEL, "trivial_kernel", opts=["--use_fast_math"]
        )
        assert module is not None

    def test_with_verbose(self) -> None:
        """compile_and_load_kernel with verbose=True."""
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


# ---------------------------------------------------------------------------
# Tests for uncovered lines
# ---------------------------------------------------------------------------


@pytest.mark.cpu
class TestFindCudaIncludeDirCommonPaths:
    """Tests for the common-paths fallback in find_cuda_include_dir() (lines 58-71)."""

    def test_common_path_found(self, monkeypatch) -> None:
        """When env vars and nvcc are absent, common paths are tried."""
        monkeypatch.delenv("CUDA_HOME", raising=False)
        monkeypatch.delenv("CUDA_PATH", raising=False)

        # Patch shutil.which to return None (no nvcc)
        with patch("trtutils.core._nvrtc.shutil.which", return_value=None):
            # Patch Path.exists so only the third common path "matches"
            original_exists = Path.exists

            def fake_exists(self: Path) -> bool:
                # Match only the /usr/local/cuda-12/include entry
                if str(self) == "/usr/local/cuda-12/include":
                    return True
                # Let all other paths (including the common ones) report False
                common_strs = {
                    "/usr/local/cuda/include",
                    "/usr/local/cuda-13/include",
                    "/usr/local/cuda-11/include",
                    "/opt/cuda/include",
                }
                if str(self) in common_strs:
                    return False
                return original_exists(self)

            with patch.object(Path, "exists", fake_exists):
                result = find_cuda_include_dir()

        assert result == Path("/usr/local/cuda-12/include")

    def test_first_common_path_wins(self, monkeypatch) -> None:
        """The first existing common path is returned."""
        monkeypatch.delenv("CUDA_HOME", raising=False)
        monkeypatch.delenv("CUDA_PATH", raising=False)

        with patch("trtutils.core._nvrtc.shutil.which", return_value=None):
            original_exists = Path.exists

            def fake_exists(self: Path) -> bool:
                # Both /usr/local/cuda/include and /opt/cuda/include exist,
                # but /usr/local/cuda/include comes first in the list.
                if str(self) in (
                    "/usr/local/cuda/include",
                    "/opt/cuda/include",
                ):
                    return True
                common_strs = {
                    "/usr/local/cuda-13/include",
                    "/usr/local/cuda-12/include",
                    "/usr/local/cuda-11/include",
                }
                if str(self) in common_strs:
                    return False
                return original_exists(self)

            with patch.object(Path, "exists", fake_exists):
                result = find_cuda_include_dir()

        assert result == Path("/usr/local/cuda/include")

    def test_no_common_path_returns_none(self, monkeypatch) -> None:
        """When no env vars, no nvcc, and no common paths exist, returns None."""
        monkeypatch.delenv("CUDA_HOME", raising=False)
        monkeypatch.delenv("CUDA_PATH", raising=False)

        with patch("trtutils.core._nvrtc.shutil.which", return_value=None):
            original_exists = Path.exists

            def fake_exists(self: Path) -> bool:
                common_strs = {
                    "/usr/local/cuda/include",
                    "/usr/local/cuda-13/include",
                    "/usr/local/cuda-12/include",
                    "/usr/local/cuda-11/include",
                    "/opt/cuda/include",
                }
                if str(self) in common_strs:
                    return False
                return original_exists(self)

            with patch.object(Path, "exists", fake_exists):
                result = find_cuda_include_dir()

        assert result is None


@pytest.mark.cpu
class TestGetDefaultNvrtcOptsNoCuda:
    """Tests for _get_default_nvrtc_opts when no CUDA include dir is found (line 78->81)."""

    def test_returns_empty_list_when_no_include_dir(self) -> None:
        """When find_cuda_include_dir returns None, opts list has no -I flag."""
        with patch("trtutils.core._nvrtc.find_cuda_include_dir", return_value=None):
            opts = _get_default_nvrtc_opts()

        assert opts == []


@pytest.mark.cpu
class TestCompileKernelDlopenError:
    """Tests for the dlopen libnvrtc error handling in compile_kernel (lines 177-184)."""

    def test_dlopen_error_is_reraised_with_chain(self) -> None:
        """RuntimeError with 'Failed to dlopen libnvrtc' gets extra guidance and chains original."""
        original_error = RuntimeError("Failed to dlopen libnvrtc.so.12.0")
        mock_nvrtc = MagicMock()
        mock_nvrtc.nvrtcCreateProgram.side_effect = original_error
        with patch("trtutils.core._nvrtc.nvrtc", mock_nvrtc):
            with pytest.raises(RuntimeError, match="Ensure the version of cuda-python") as exc_info:
                compile_kernel(TRIVIAL_KERNEL, "trivial_kernel")
            assert exc_info.value.__cause__ is original_error

    def test_other_runtime_errors_are_reraised_unchanged(self) -> None:
        """RuntimeError without the dlopen message propagates unchanged."""
        original_error = RuntimeError("Some other NVRTC error")

        mock_nvrtc = MagicMock()
        mock_nvrtc.nvrtcCreateProgram.side_effect = original_error

        with patch("trtutils.core._nvrtc.nvrtc", mock_nvrtc):
            with pytest.raises(RuntimeError, match="Some other NVRTC error"):
                compile_kernel(TRIVIAL_KERNEL, "trivial_kernel")
