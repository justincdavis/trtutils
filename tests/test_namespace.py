# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
import trtutils


def test_namespace():
    assert trtutils.TRTEngine is not None
    assert trtutils.TRTModel is not None
