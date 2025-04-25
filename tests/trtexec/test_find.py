# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import trtutils


def test_find_trtexec() -> None:
    """Test if the find trtexec utility works."""
    if not trtutils.FLAGS.IS_JETSON:
        return

    # otherwise we should get a valid path from find trtexec
    path = trtutils.trtexec.find_trtexec()

    assert path.exists()
