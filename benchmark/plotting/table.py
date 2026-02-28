# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: T201
"""Generate RST tables of benchmarking results."""

from __future__ import annotations

import json
from pathlib import Path

from utils.config import DATA_DIR, MODEL_FRAMEWORKS

_RST_DIR = Path(__file__).resolve().parent.parent / "rst"


def format_value(value: float | str) -> str:
    """Format a metric value for display in a table."""
    return f"{value:.1f}" if isinstance(value, float) else "N/A"


def _make_rst_table(rows: list[list[str]], headers: list[str]) -> str:
    """Create an RST csv-table from rows and headers."""
    table = ".. csv-table:: Performance Metrics\n"
    table += "   :header: " + ",".join(headers) + "\n"
    table += "   :widths: " + ",".join(["10"] * len(headers)) + "\n\n"

    for row in rows:
        table += "   " + ",".join(str(cell) for cell in row) + "\n"

    return table


def _get_device_rows(
    data: dict[str, dict[str, dict[str, dict[str, float]]]],
) -> list[list[str]]:
    """Extract rows for a single device's data."""
    rows: list[list[str]] = []
    for framework in reversed(MODEL_FRAMEWORKS):
        f_data = data.get(framework)
        if f_data is None:
            continue
        for modelname, m_data in f_data.items():
            for input_size, metrics in m_data.items():
                rows.append([
                    framework,
                    modelname,
                    input_size,
                    format_value(metrics.get("mean", "N/A")),
                    format_value(metrics.get("median", "N/A")),
                    format_value(metrics.get("min", "N/A")),
                    format_value(metrics.get("max", "N/A")),
                ])
    return rows


def _make_all_table(
    data: dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]],
) -> str:
    """Create a combined RST table for all devices."""
    rows: list[list[str]] = []
    for device, d_data in data.items():
        for r in _get_device_rows(d_data):
            rows.append([device, *r])

    headers = [
        "Device", "Framework", "Model", "Input Size",
        "Mean (ms)", "Median (ms)", "Min (ms)", "Max (ms)",
    ]
    return _make_rst_table(rows, headers)


def _make_device_table(
    data: dict[str, dict[str, dict[str, dict[str, float]]]],
) -> str:
    """Create an RST table for a single device."""
    rows = _get_device_rows(data)
    headers = [
        "Framework", "Model", "Input Size",
        "Mean (ms)", "Median (ms)", "Min (ms)", "Max (ms)",
    ]
    return _make_rst_table(rows, headers)


def generate_tables(
    *,
    skip_devices: set[str] | None = None,
) -> None:
    """Generate RST tables from benchmark data."""
    skip = skip_devices or set()
    data_dir = DATA_DIR / "models"

    all_data: dict[str, dict] = {}
    for file in data_dir.iterdir():
        if file.is_dir() or file.suffix != ".json":
            continue
        name = file.stem
        if name in skip:
            print(f"Skipping device: {name}")
            continue
        try:
            with file.open("r") as f:
                all_data[name] = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load data for {name}: {e}, skipping...")
            continue

    _RST_DIR.mkdir(parents=True, exist_ok=True)

    # Overall table
    print("Making overall table...")
    all_table = _make_all_table(all_data)
    with (_RST_DIR / "overview.rst").open("w") as f:
        f.write(all_table)

    # Per-device tables
    device_dir = _RST_DIR / "devices"
    device_dir.mkdir(parents=True, exist_ok=True)
    for device, device_data in all_data.items():
        print(f"Making {device} table...")
        device_table = _make_device_table(device_data)
        with (device_dir / f"{device}.rst").open("w") as f:
            f.write(device_table)
