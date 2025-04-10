# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from trtutils.core import Kernel, init_cuda


def kernel_compile(kernel: tuple[str, str]) -> None:
    """
    Test if a kernel will compile.
    
    Parameters
    ----------
    kernel : tuple[str, str]
        The kernel info
    
    """
    def _compile() -> None:
        compiled = Kernel(*kernel)
        assert compiled is not None

    # if we get a CUDA not initialized error then we can init and try again
    try:
        _compile()
    except RuntimeError as err:
        if "Initialized" in err:
            init_cuda()
            _compile()
        else:
            raise err
