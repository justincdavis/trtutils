# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path

import trtutils


def test_find_trtexec() -> None:
    """Test if the find trtexec utility works."""
    # if we are on a jetson system the /etc/nv_tegra_release file exists
    tegra = Path("/etc/nv_tegra_release")
    if not tegra.exists():
        return
    
    # otherwise we should get a valid path from find trtexec
    path = trtutils.trtexec.find_trtexec()

    assert path.exists()
