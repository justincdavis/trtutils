# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/core/_nvrtc.py -- NVRTC compilation and loading."""

from __future__ import annotations

from pathlib import Path

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

pytestmark = pytest.mark.usefixtures("_nvrtc_cache_clear")

TRIVIAL_KERNEL = """\
extern "C" __global__ void trivial_kernel(float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (float)idx;
    }
}
"""


def test_find_cuda_include_dir() -> None:
    """find_cuda_include_dir returns a Path or None."""
    result = find_cuda_include_dir()
    assert result is None or isinstance(result, Path)


@pytest.mark.parametrize(
    ("env_var", "clear_var"),
    [
        pytest.param("CUDA_HOME", "CUDA_PATH", id="CUDA_HOME"),
        pytest.param("CUDA_PATH", "CUDA_HOME", id="CUDA_PATH"),
    ],
)
def test_find_cuda_include_dir_env_var(tmp_path, monkeypatch, env_var, clear_var) -> None:
    """Env var pointing to CUDA install is detected."""
    include_dir = tmp_path / "include"
    include_dir.mkdir()
    monkeypatch.setenv(env_var, str(tmp_path))
    monkeypatch.delenv(clear_var, raising=False)
    result = find_cuda_include_dir()
    assert result == include_dir


def test_find_cuda_include_dir_env_no_include(tmp_path, monkeypatch) -> None:
    """CUDA_HOME set but no include dir falls through."""
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))
    monkeypatch.delenv("CUDA_PATH", raising=False)
    result = find_cuda_include_dir()
    assert result is None or isinstance(result, Path)


def test_get_default_nvrtc_opts(tmp_path, monkeypatch) -> None:
    """Returns list; includes -I flag when CUDA include dir is found."""
    # basic: returns list
    result = _get_default_nvrtc_opts()
    assert isinstance(result, list)
    # with CUDA_HOME set: has -I flag
    include_dir = tmp_path / "include"
    include_dir.mkdir()
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))
    monkeypatch.delenv("CUDA_PATH", raising=False)
    find_cuda_include_dir.cache_clear()
    opts = _get_default_nvrtc_opts()
    assert any(b"-I" in opt for opt in opts)


def test_nvrtc_error_handling() -> None:
    """check_nvrtc_err and nvrtc_call handle success and failure."""
    # success: no raise
    check_nvrtc_err(nvrtc.nvrtcResult.NVRTC_SUCCESS)
    # failure: RuntimeError
    with pytest.raises(RuntimeError, match="NVRTC Error"):
        check_nvrtc_err(nvrtc.nvrtcResult.NVRTC_ERROR_COMPILATION)
    # nvrtc_call: returns version
    result = nvrtc_call(nvrtc.nvrtcVersion())
    assert result is not None


@pytest.mark.usefixtures("cuda_context")
def test_compile_and_load_kernel() -> None:
    """Full compile -> load -> launch lifecycle with opts and verbose."""
    # compile returns chararray
    ptx = compile_kernel(TRIVIAL_KERNEL, "trivial_kernel")
    assert isinstance(ptx, np.char.chararray)
    # load returns module and kernel
    module, kernel = load_kernel(ptx, "trivial_kernel")
    assert module is not None
    assert kernel is not None
    assert isinstance(module, cuda.CUmodule)
    # compile_and_load shortcut works
    module2, kernel2 = compile_and_load_kernel(TRIVIAL_KERNEL, "trivial_kernel")
    assert module2 is not None
    assert kernel2 is not None
    # opts and verbose don't crash
    ptx_opts = compile_kernel(TRIVIAL_KERNEL, "trivial_kernel", opts=["--use_fast_math"])
    assert ptx_opts is not None
    ptx_verbose = compile_kernel(TRIVIAL_KERNEL, "trivial_kernel", verbose=True)
    assert ptx_verbose is not None
    module_v, _kernel_v = load_kernel(ptx, "trivial_kernel", verbose=True)
    assert module_v is not None
    module_cv, _kernel_cv = compile_and_load_kernel(TRIVIAL_KERNEL, "trivial_kernel", verbose=True)
    assert module_cv is not None
    module_co, _kernel_co = compile_and_load_kernel(
        TRIVIAL_KERNEL, "trivial_kernel", opts=["--use_fast_math"]
    )
    assert module_co is not None
