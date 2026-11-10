# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/download/_tools.py -- handle_batch."""

from __future__ import annotations

import subprocess

import pytest

from trtutils.download._tools import (
    get_patches_dir,
    get_repo_cache_dir,
    handle_batch,
    run_patch,
)

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize(
    ("batch", "dynamic", "supported", "expected"),
    [
        pytest.param(None, None, True, (1, False), id="defaults"),
        pytest.param(None, None, False, (1, False), id="defaults-unsupported"),
        pytest.param(1, False, False, (1, False), id="batch-1-unsupported"),
        pytest.param(8, None, True, (8, False), id="batch-8"),
        pytest.param(8, False, True, (8, False), id="batch-8-static"),
        pytest.param(None, True, True, (1, True), id="dynamic"),
        pytest.param(1, True, True, (1, True), id="dynamic-batch-1"),
    ],
)
def test_handle_batch(batch, dynamic, supported, expected) -> None:
    """None means batch 1 / static; supported exporters accept any batch or dynamic."""
    assert handle_batch(batch, "fam", dynamic=dynamic, supported=supported) == expected


@pytest.mark.parametrize(
    ("batch", "dynamic", "supported", "error", "match"),
    [
        pytest.param(0, None, True, ValueError, "positive", id="zero"),
        pytest.param(-1, None, True, ValueError, "positive", id="negative"),
        pytest.param(0, None, False, ValueError, "positive", id="zero-unsupported"),
        pytest.param(4, True, True, ValueError, "fam", id="dynamic-with-batch"),
        pytest.param(2, None, False, NotImplementedError, "fam", id="batch-unsupported"),
        pytest.param(None, True, False, NotImplementedError, "fam", id="dynamic-unsupported"),
        pytest.param(4, True, False, NotImplementedError, "fam", id="both-unsupported"),
    ],
)
def test_handle_batch_rejects(batch, dynamic, supported, error, match) -> None:
    """Non-positive batches, dynamic plus a batch, or anything above 1 on an unsupported exporter."""
    with pytest.raises(error, match=match):
        handle_batch(batch, "fam", dynamic=dynamic, supported=supported)


_DEIMV2_PIN = "19d5b19a58c229dd7ad5f079947bbe398e005d01"


def test_deimv2_patch_exports_batch_one_without_grad(tmp_path) -> None:
    """The DEIMv2 patch applies to the pinned script, traces batch 2 and disables autograd."""
    repo = get_repo_cache_dir() / "DEIMv2"
    if not repo.exists():
        pytest.skip("DEIMv2 repo is not cached")
    script = tmp_path / "export_onnx.py"
    shown = subprocess.run(  # noqa: S603
        ["git", "-C", str(repo), "show", f"{_DEIMV2_PIN}:tools/deployment/export_onnx.py"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    if shown.returncode != 0:
        pytest.skip("pinned DEIMv2 commit is not cached")
    script.write_text(shown.stdout)
    run_patch(tmp_path, str(get_patches_dir() / "deimv2_export_onnx.patch"), script.name)
    patched = script.read_text()
    assert "torch.rand(2, 3, *img_size)" in patched
    assert "torch.rand(32" not in patched
    assert "torch.set_grad_enabled(False)" in patched
