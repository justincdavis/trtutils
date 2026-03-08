# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from trtutils._log import LOG

if TYPE_CHECKING:
    from trtutils.builder._batcher import AbstractBatcher


def generate_calibration_data(
    batcher: AbstractBatcher,
    output_path: Path | str,
    *,
    verbose: bool | None = None,
) -> Path:
    """
    Generate calibration data from a batcher and save to a .npy file.

    Parameters
    ----------
    batcher : AbstractBatcher
        A batcher instance to drain batches from.
    output_path : Path, str
        The path to save the calibration data to.
    verbose : bool, optional
        Whether to print verbose output, by default None.

    Returns
    -------
    Path
        The resolved path to the saved calibration data.

    Raises
    ------
    ValueError
        If no batches could be retrieved from the batcher.

    """
    batches: list[np.ndarray] = []
    batch_idx = 0

    while True:
        batch = batcher.get_next_batch()
        if batch is None:
            break
        batches.append(batch)
        batch_idx += 1
        if verbose:
            LOG.debug(f"Collected calibration batch {batch_idx}")

    if len(batches) == 0:
        err_msg = "No batches could be retrieved from the batcher."
        raise ValueError(err_msg)

    calibration_data = np.concatenate(batches, axis=0)

    if verbose:
        LOG.debug(
            f"Calibration data shape: {calibration_data.shape}, "
            f"dtype: {calibration_data.dtype}"
        )

    output = Path(output_path).resolve()
    np.save(output, calibration_data)

    LOG.info(f"Saved calibration data to {output}")

    return output
