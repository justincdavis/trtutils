# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

from typing import TYPE_CHECKING

from trtutils._flags import FLAGS

if TYPE_CHECKING:
    from collections.abc import Callable

    from trtutils.compat._libs import trt


def get_check_dla(config: trt.IBuilderConfig) -> Callable[[trt.ILayer], bool]:
    """
    Get the check_dla function for the given config.

    Parameters
    ----------
    config : trt.IBuilderConfig
        The TensorRT builder config.

    Returns
    -------
    Callable[[trt.ILayer], bool]
        The check_dla function.

    """
    check_dla: Callable[[trt.ILayer], bool] = (
        config.can_run_on_DLA if FLAGS.NEW_CAN_RUN_ON_DLA else config.canRunOnDLA
    )
    return check_dla
