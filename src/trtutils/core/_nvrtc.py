# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib
import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import TypeVar

import numpy as np

with contextlib.suppress(Exception):
    try:
        import cuda.bindings.driver as cuda
        import cuda.bindings.nvrtc as nvrtc
    except (ImportError, ModuleNotFoundError):
        from cuda import cuda, nvrtc

from trtutils._log import LOG

from ._cuda import cuda_call
from ._lock import MEM_ALLOC_LOCK, NVRTC_LOCK


@lru_cache(maxsize=1)
def find_cuda_include_dir() -> Path | None:
    """
    Find the CUDA include directory for NVRTC compilation.

    Searches in the following order:
    1. CUDA_HOME environment variable
    2. CUDA_PATH environment variable
    3. Path derived from nvcc location
    4. Common default paths (/usr/local/cuda, etc.)

    Returns
    -------
    Path | None
        The path to the CUDA include directory, or None if not found.

    """
    for env_var in ("CUDA_HOME", "CUDA_PATH"):
        cuda_path = os.environ.get(env_var)
        if cuda_path:
            include_path = Path(cuda_path) / "include"
            if include_path.exists():
                LOG.debug(f"Found CUDA include path from {env_var}: {include_path}")
                return include_path

    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        cuda_root = Path(nvcc_path).parent.parent
        include_path = cuda_root / "include"
        if include_path.exists():
            LOG.debug(f"Found CUDA include path from nvcc: {include_path}")
            return include_path

    common_paths = [
        Path("/usr/local/cuda/include"),
        Path("/usr/local/cuda-13/include"),
        Path("/usr/local/cuda-12/include"),
        Path("/usr/local/cuda-11/include"),
        Path("/opt/cuda/include"),
    ]
    for include_path in common_paths:
        if include_path.exists():
            LOG.debug(f"Found CUDA include path at default location: {include_path}")
            return include_path

    LOG.warning("Could not find CUDA include path for NVRTC compilation")
    return None


def _get_default_nvrtc_opts() -> list[bytes]:
    opts: list[bytes] = []

    include_path = find_cuda_include_dir()
    if include_path:
        opts.append(f"-I{include_path}".encode())

    return opts


def check_nvrtc_err(err: nvrtc.nvrtcResult) -> None:
    """
    Check if a NVRTC error occured and raise an exception if so.

    Parameters
    ----------
    err : nvrtc.nvrtcResult
        The NVRTC return code to check.

    Raises
    ------
    RuntimeError
        If a NVRTC error occured.

    """
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        err_msg = f"NVRTC Error: {err} -> "
        err_msg += f"{nvrtc_call(nvrtc.nvrtcGetErrorString(err))}"
        raise RuntimeError(err_msg)


T = TypeVar("T")


def nvrtc_call(call: tuple[nvrtc.nvrtcResult, T]) -> T:
    """
    Call a NVRTC function and check for errors.

    Parameters
    ----------
    call : tuple[cuda.CUresult | cudart.cudaError_t, T]
        The NVRTC function to call and its arguments.

    Returns
    -------
    T
        The result of the NVRTC function call.

    """
    err, res = call[0], call[1:]
    check_nvrtc_err(err)
    if len(res) == 1:
        return res[0]
    return res


def compile_kernel(
    kernel: str,
    name: str,
    opts: list[str] | None = None,
    *,
    verbose: bool | None = None,
) -> np.char.chararray:
    """
    Compile a CUDA kernel into PTX using NVRTC.

    Parameters
    ----------
    kernel : str
        The kernel definition in CUDA.
    name : str
        The name of the kernel in the definition.
    opts : list[str]
        The optional additional arguments to pass to NVRTC during
        the compilation of the kernel.
    verbose : bool, optional
        Whether or not to output additional information
        to stdout. If not provided, will default to overall
        engines verbose setting.

    Returns
    -------
    tuple[np.char.chararray, str]
        The compiled PTX kernel and the kernel name.

    Raises
    ------
    RuntimeError
        If the version of cuda-python installed does not match the version of CUDA installed.

    """
    kernel_bytes = kernel.encode()
    kernel_name_bytes = f"{name}.cu".encode()

    if verbose:
        LOG.debug(f"Compiling kernel: {name}")

    # compile the kernel
    try:
        with MEM_ALLOC_LOCK, NVRTC_LOCK:
            prog = nvrtc_call(
                nvrtc.nvrtcCreateProgram(kernel_bytes, kernel_name_bytes, 0, [], []),
            )
    except RuntimeError as err:
        if "Failed to dlopen libnvrtc" in str(err):
            err_msg = str(err)
            err_msg += (
                " Ensure the version of cuda-python installed matches the version of CUDA installed."
            )
            raise RuntimeError(err_msg) from err
        raise

    opts_with_cuda_include_dir = _get_default_nvrtc_opts()
    if opts is not None:
        for opt in opts:
            opts_with_cuda_include_dir.append(opt.encode())

    with MEM_ALLOC_LOCK, NVRTC_LOCK:
        nvrtc_call(nvrtc.nvrtcCompileProgram(prog, len(opts_with_cuda_include_dir), opts_with_cuda_include_dir))

    # generate the actual kernel ptx
    ptx_size = nvrtc_call(nvrtc.nvrtcGetPTXSize(prog))
    ptx_buffer = b"\0" * ptx_size
    nvrtc_call(nvrtc.nvrtcGetPTX(prog, ptx_buffer))

    return np.char.array(ptx_buffer)


def load_kernel(
    kernel_ptx: np.char.chararray,
    name: str,
    *,
    verbose: bool | None = None,
) -> tuple[cuda.CUmodule, cuda.CUkernel]:
    """
    Load a kernel from a PTX definition.

    Parameters
    ----------
    kernel_ptx: np.char.chararray
        The PTX generated by NVRTC, use the compile_kernel function.
    name: str
        The name of the kernel inside the PTX definiton.
    verbose : bool, optional
        Whether or not to output additional information
        to stdout. If not provided, will default to overall
        engines verbose setting.

    Returns
    -------
    tuple[cuda.CUmodule, cuda.CUkernel]
        The CUDA module and kernel

    """
    if verbose:
        LOG.debug(f"Loading kernel: {name} from PTX")

    module: cuda.CUmodule = cuda_call(cuda.cuModuleLoadData(kernel_ptx.ctypes.data))
    kernel: cuda.CUkernel = cuda_call(
        cuda.cuModuleGetFunction(module, name.encode()),
    )
    return module, kernel


def compile_and_load_kernel(
    kernel_code: str,
    name: str,
    opts: list[str] | None = None,
    *,
    verbose: bool | None = None,
) -> tuple[cuda.CUmodule, cuda.CUkernel]:
    """
    Compile and load a kernel from source definiton.

    Parameters
    ----------
    kernel_code : str
        The code definition of the kernel.
    name : str
        The name of the kernel.
    opts : list[str]
        The optional additional arguments to pass to NVRTC during
        the compilation of the kernel.
    verbose : bool, optional
        Whether or not to output additional information
        to stdout. If not provided, will default to overall
        engines verbose setting.

    Returns
    -------
    tuple[cuda.CUmodule, cuda.CUkernel]
        The CUDA module and kernel

    """
    ptx = compile_kernel(kernel_code, name, opts, verbose=verbose)
    module, kernel = load_kernel(ptx, name, verbose=verbose)
    return module, kernel
