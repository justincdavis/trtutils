# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for MANIFEST.in and package-data -- installed copies ship every CUDA kernel file."""

from __future__ import annotations

import re
import sys
from fnmatch import fnmatch
from pathlib import Path

import pytest

from trtutils.image import kernels

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

pytestmark = pytest.mark.cpu

_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE = _ROOT / "src" / "trtutils"
_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)


def test_kernel_includes_resolve() -> None:
    """Every #include "..." in the imported kernels names a file in their directory."""
    kernel_dir = kernels.YOLO_NMS.parent
    includes = {
        (src.name, name)
        for src in kernel_dir.iterdir()
        if src.suffix in (".cu", ".cuh")
        for name in _INCLUDE.findall(src.read_text())
    }
    assert ("yolo_nms.cu", "postprocess.cuh") in includes
    missing = sorted((src, name) for src, name in includes if not (kernel_dir / name).is_file())
    assert missing == []


def test_kernel_files_are_packaged() -> None:
    """Every file in image/_kernels matches a MANIFEST.in include and a package-data glob."""
    manifest = [
        pattern
        for line in (_ROOT / "MANIFEST.in").read_text().splitlines()
        if line.startswith("include ")
        for pattern in line.split()[1:]
    ]
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    package_data = pyproject["tool"]["setuptools"]["package-data"]["trtutils"]
    files = [p for p in (_PACKAGE / "image" / "_kernels").iterdir() if p.is_file()]
    assert any(p.suffix == ".cuh" for p in files)
    for path in files:
        from_root = path.relative_to(_ROOT).as_posix()
        from_package = path.relative_to(_PACKAGE).as_posix()
        assert any(fnmatch(from_root, pattern) for pattern in manifest), from_root
        assert any(fnmatch(from_package, pattern) for pattern in package_data), from_package
