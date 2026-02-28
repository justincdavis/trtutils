# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Data I/O utilities for benchmark results."""

from __future__ import annotations

import json

from .config import DATA_DIR


def get_data(
    device: str,
    data_subdir: str,
    frameworks: list[str],
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Load benchmark data for a device.

    Parameters
    ----------
    device : str
        Device name (e.g. "5080").
    data_subdir : str
        Subdirectory under data/ (e.g. "models", "batch").
    frameworks : list[str]
        Framework keys to ensure exist in the data.

    Returns
    -------
    dict
        Nested dict: framework -> model -> size/batch -> metrics.

    """
    subdir = DATA_DIR / data_subdir
    subdir.mkdir(parents=True, exist_ok=True)
    file_path = subdir / f"{device}.json"

    if file_path.exists():
        with file_path.open("r") as f:
            data = json.load(f)
        for fw in frameworks:
            if data.get(fw) is None:
                data[fw] = {}
        return data

    return {fw: {} for fw in frameworks}


def write_data(
    device: str,
    data_subdir: str,
    data: dict[str, dict[str, dict[str, dict[str, float]]]],
) -> None:
    """Write benchmark data for a device.

    Parameters
    ----------
    device : str
        Device name.
    data_subdir : str
        Subdirectory under data/ (e.g. "models", "batch").
    data : dict
        Benchmark data to write.

    """
    subdir = DATA_DIR / data_subdir
    subdir.mkdir(parents=True, exist_ok=True)
    file_path = subdir / f"{device}.json"
    with file_path.open("w") as f:
        json.dump(data, f, indent=4)
