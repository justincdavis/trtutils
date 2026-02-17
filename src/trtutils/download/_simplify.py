# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: S603
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from trtutils._log import LOG

from ._tools import check_uv_version, get_tool_requirements, make_venv, run_cmd, run_uv_pip_install

_TOOLS: list[tuple[str, str]] = [
    ("onnxsim", "cli"),
    ("onnxslim", "cli"),
    ("polygraphy", "surgeon"),
]


def simplify(
    model_path: Path,
    *,
    directory: Path | None = None,
    bin_path: Path | None = None,
    no_uv_cache: bool | None = None,
    verbose: bool | None = None,
) -> None:
    """
    Simplify an ONNX model using multiple tools and keep the best result.

    Runs onnxsim, onnxslim, and polygraphy, compares their outputs by
    graph node count, and keeps the simplest result.

    Parameters
    ----------
    model_path : Path
        The path to the ONNX model to simplify.
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
        _run_simplify(model_path, directory, bin_path, no_uv_cache=no_uv_cache, verbose=verbose)
    else:
        check_uv_version()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            _, venv_bin_path = make_venv(temp_path, no_cache=no_uv_cache, verbose=verbose)
            _run_simplify(
                model_path, temp_path, venv_bin_path, no_uv_cache=no_uv_cache, verbose=verbose
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
            "-o",
            str(output_path),
        ]
    return [bin_path / tool_name, str(input_path), str(output_path)]


def _count_nodes(python_path: Path, model_path: Path) -> int | None:
    script = "import onnx; m = onnx.load(r'" + str(model_path) + "'); print(len(m.graph.node))"
    try:
        result = subprocess.run(
            [str(python_path), "-c", script],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError, OSError):
        return None


def _run_tool(
    tool_name: str,
    tool_type: str,
    model_path: Path,
    output_path: Path,
    directory: Path,
    bin_path: Path,
    *,
    no_uv_cache: bool | None = None,
    verbose: bool | None = None,
) -> int | None:
    try:
        LOG.info(f"Installing {tool_name} into venv at {directory}")
        run_uv_pip_install(
            directory,
            bin_path.parent,
            model=None,
            packages=["-r", get_tool_requirements(tool_name)],
            no_cache=no_uv_cache,
            verbose=verbose,
        )
        LOG.info(f"Running {tool_name} on {model_path.stem}")
        cmd = _build_tool_cmd(tool_name, tool_type, bin_path, model_path, output_path)
        run_cmd(cmd, verbose=verbose)
        python_path = bin_path / "python"
        node_count = _count_nodes(python_path, output_path)
        if node_count is not None:
            LOG.info(f"{tool_name} produced {node_count} nodes")
        else:
            LOG.warning(f"{tool_name} completed but could not count nodes in output")
    except (subprocess.CalledProcessError, OSError, RuntimeError):
        LOG.warning(f"{tool_name} failed, skipping")
        if output_path.exists():
            output_path.unlink()
        return None
    else:
        return node_count


def _run_simplify(
    model_path: Path,
    directory: Path,
    bin_path: Path,
    *,
    no_uv_cache: bool | None = None,
    verbose: bool | None = None,
) -> None:
    LOG.info(f"Simplifying ONNX model: {model_path.stem}")

    best_count: int | None = None
    best_path: Path | None = None
    outputs: list[Path] = []

    for tool_name, tool_type in _TOOLS:
        output_path = model_path.with_suffix(f".{tool_name}.onnx")
        outputs.append(output_path)

        node_count = _run_tool(
            tool_name,
            tool_type,
            model_path,
            output_path,
            directory,
            bin_path,
            no_uv_cache=no_uv_cache,
            verbose=verbose,
        )

        if node_count is not None and (best_count is None or node_count < best_count):
            # Clean up previous best if it exists
            if best_path is not None and best_path.exists():
                best_path.unlink()
            best_count = node_count
            best_path = output_path
        elif output_path.exists():
            # Not the best, clean up
            output_path.unlink()

    if best_path is None or best_count is None:
        # Clean up any remaining outputs
        for output_path in outputs:
            if output_path.exists():
                output_path.unlink()
        err_msg = "All simplification tools failed"
        raise RuntimeError(err_msg)

    # Count original nodes for comparison
    python_path = bin_path / "python"
    original_count = _count_nodes(python_path, model_path)
    if original_count is not None:
        LOG.info(
            f"Original: {original_count} nodes -> Best ({best_path.suffixes[-2][1:]}): {best_count} nodes"
        )
    else:
        LOG.info(f"Best result from {best_path.suffixes[-2][1:]}: {best_count} nodes")

    shutil.move(str(best_path), str(model_path))
    LOG.info(f"Simplified ONNX model: {model_path.stem}")
