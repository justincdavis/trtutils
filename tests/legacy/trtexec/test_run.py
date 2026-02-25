# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import trtutils


def test_run_trtexec() -> None:
    """Test if the run trtexec utility works."""
    if not trtutils.FLAGS.IS_JETSON:
        return

    success, _, _ = trtutils.trtexec.run_trtexec("--help")

    assert success
