# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import tempfile
import textwrap
from pathlib import Path

from trtutils._log import LOG

from ._tools import check_uv_version, get_tool_requirements, make_venv, run_cmd, run_uv_pip_install


def make_static(
    model_path: Path,
    *,
    directory: Path | None = None,
    bin_path: Path | None = None,
    no_uv_cache: bool | None = None,
    verbose: bool | None = None,
) -> None:
    """
    Set any dynamic dimensions in an ONNX model to fixed values.

    Replaces all dynamic dimensions (symbolic or unset) with a static
    value of 1. This is useful when a model must have fully static
    shapes, for example when targeting hardware accelerators that do
    not support dynamic shapes.

    Parameters
    ----------
    model_path : Path
        The path to the ONNX model to make static.
    directory : Path, optional
        The working directory containing an existing venv.
        If None, a temporary directory with a fresh venv will be created.
    bin_path : Path, optional
        The bin directory of an existing venv.
        If None, a temporary venv will be created.
    no_uv_cache : bool, optional
        Whether to disable caching of uv packages.
    verbose : bool, optional
        Whether to print verbose output.

    """
    if directory is not None and bin_path is not None:
        _run_make_static(
            model_path,
            directory,
            bin_path,
            no_uv_cache=no_uv_cache,
            verbose=verbose,
        )
    else:
        check_uv_version()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            _, venv_bin_path = make_venv(temp_path, no_cache=no_uv_cache, verbose=verbose)
            _run_make_static(
                model_path,
                temp_path,
                venv_bin_path,
                no_uv_cache=no_uv_cache,
                verbose=verbose,
            )


_MAKE_STATIC_SCRIPT = textwrap.dedent("""\
    import sys
    import onnx

    model_path = sys.argv[1]
    model = onnx.load(model_path)
    changed = False
    for inp in model.graph.input:
        for dim in inp.type.tensor_type.shape.dim:
            if dim.dim_value <= 0:
                dim.ClearField("dim_param")
                dim.dim_value = 1
                changed = True
    if changed:
        onnx.save(model, model_path)
""")


def _run_make_static(
    model_path: Path,
    directory: Path,
    bin_path: Path,
    *,
    no_uv_cache: bool | None = None,
    verbose: bool | None = None,
) -> None:
    LOG.info(f"Making ONNX model static: {model_path.stem}")

    LOG.info("Installing onnx tools into venv")
    run_uv_pip_install(
        directory,
        bin_path.parent,
        model=None,
        packages=["-r", get_tool_requirements("onnx")],
        no_cache=no_uv_cache,
        verbose=verbose,
    )

    cmd: list[str | Path] = [
        bin_path / "python",
        "-c",
        _MAKE_STATIC_SCRIPT,
        str(model_path),
    ]
    run_cmd(cmd, verbose=verbose)

    LOG.info(f"Made ONNX model static: {model_path.stem}")
