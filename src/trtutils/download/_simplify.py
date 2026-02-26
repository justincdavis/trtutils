# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

from trtutils._log import LOG

from ._tools import check_uv_version, get_tool_requirements, make_venv, run_cmd, run_uv_pip_install

_TOOL_REGISTRY: dict[str, str] = {
    "polygraphy": "surgeon",
    "onnxslim": "cli",
    "onnxsim": "cli",
}
_VALID_TOOLS: frozenset[str] = frozenset(_TOOL_REGISTRY.keys())
_DEFAULT_TOOLS: list[str] = ["polygraphy", "onnxslim"]


def simplify(
    model_path: Path,
    *,
    tools: Sequence[str] | None = None,
    directory: Path | None = None,
    bin_path: Path | None = None,
    no_uv_cache: bool | None = None,
    verbose: bool | None = None,
) -> None:
    """
    Simplify an ONNX model by running tools as a sequential pipeline.

    Runs the specified simplification tools in sequence, where each tool's
    output becomes the next tool's input. If any tool fails, it is skipped
    and the current result is passed to the next tool.

    Parameters
    ----------
    model_path : Path
        The path to the ONNX model to simplify.
    tools : Sequence[str], optional
        Which simplification tools to run and in what order.
        Valid tool names: "polygraphy", "onnxslim", "onnxsim".
        If None, uses the default tools (polygraphy, onnxslim).
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

    Raises
    ------
    ValueError
        If any tool name is not recognized.

    """
    resolved_tools = list(tools) if tools is not None else list(_DEFAULT_TOOLS)
    unknown = set(resolved_tools) - _VALID_TOOLS
    if unknown:
        err_msg = f"Unknown simplification tools: {', '.join(sorted(unknown))}. Valid tools: {', '.join(sorted(_VALID_TOOLS))}"
        raise ValueError(err_msg)

    if directory is not None and bin_path is not None:
        _run_simplify(
            model_path,
            directory,
            bin_path,
            resolved_tools,
            no_uv_cache=no_uv_cache,
            verbose=verbose,
        )
    else:
        check_uv_version()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            _, venv_bin_path = make_venv(temp_path, no_cache=no_uv_cache, verbose=verbose)
            _run_simplify(
                model_path,
                temp_path,
                venv_bin_path,
                resolved_tools,
                no_uv_cache=no_uv_cache,
                verbose=verbose,
            )


def _build_tool_cmd(
    tool_name: str,
    tool_type: str,
    bin_path: Path,
    input_path: Path,
    output_path: Path,
) -> list[str | Path]:
    if tool_type == "surgeon":
        return [
            bin_path / tool_name,
            "surgeon",
            "sanitize",
            str(input_path),
            "--fold-constants",
            "-o",
            str(output_path),
        ]
    return [bin_path / tool_name, str(input_path), str(output_path)]


def _run_simplify(
    model_path: Path,
    directory: Path,
    bin_path: Path,
    tools: list[str],
    *,
    no_uv_cache: bool | None = None,
    verbose: bool | None = None,
) -> None:
    LOG.info(f"Simplifying ONNX model: {model_path.stem}")

    LOG.info("Installing simplification tools into venv")
    run_uv_pip_install(
        directory,
        bin_path.parent,
        model=None,
        packages=["-r", get_tool_requirements("simplify")],
        no_cache=no_uv_cache,
        verbose=verbose,
    )

    current_input = model_path
    intermediates: list[Path] = []

    for tool_name in tools:
        tool_type = _TOOL_REGISTRY[tool_name]
        output_path = model_path.with_suffix(f".{tool_name}.onnx")
        try:
            LOG.info(f"Running {tool_name} on {current_input.stem}")
            cmd = _build_tool_cmd(tool_name, tool_type, bin_path, current_input, output_path)
            run_cmd(cmd, verbose=verbose)
            intermediates.append(output_path)
            current_input = output_path
        except subprocess.CalledProcessError:
            LOG.error(f"{tool_name} failed, skipping")
            if output_path.exists():
                output_path.unlink()

    if current_input == model_path:
        for path in intermediates:
            if path.exists():
                path.unlink()
        err_msg = "All simplification tools failed"
        raise RuntimeError(err_msg)

    shutil.move(str(current_input), str(model_path))

    for path in intermediates:
        if path.exists():
            path.unlink()

    LOG.info(f"Simplified ONNX model: {model_path.stem}")
