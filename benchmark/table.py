# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Generate RST tables of benchmarking results."""

from __future__ import annotations

import json
from pathlib import Path

from run import FRAMEWORKS


def format_value(value: float | str) -> str:
    return f"{value:.1f}" if isinstance(value, float) else "N/A"


def _make_rst_table(rows: list[list[str]], headers: list[str]) -> str:
    # Create the RST table header
    table = ".. csv-table:: Performance Metrics\n"
    table += "   :header: " + ",".join(headers) + "\n"
    table += "   :widths: " + ",".join(["10"] * len(headers)) + "\n\n"

    # Add the rows
    for row in rows:
        table += "   " + ",".join(str(cell) for cell in row) + "\n"

    return table


def _get_device_rows(
    data: dict[str, dict[str, dict[str, dict[str, float]]]],
) -> list[list[str]]:
    rows = []
    for framework in reversed(FRAMEWORKS):
        f_data = data.get(framework)
        if f_data is None:
            continue
        for modelname, m_data in f_data.items():
            for input_size, metrics in m_data.items():
                mean = format_value(metrics.get("mean", "N/A"))
                median = format_value(metrics.get("median", "N/A"))
                min_val = format_value(metrics.get("min", "N/A"))
                max_val = format_value(metrics.get("max", "N/A"))

                rows.append(
                    [
                        framework,
                        modelname,
                        input_size,
                        mean,
                        median,
                        min_val,
                        max_val,
                    ],
                )
    return rows


def _make_all_table(
    data: dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]],
) -> str:
    rows = []
    for device, d_data in data.items():
        device_rows = _get_device_rows(d_data)
        for r in device_rows:
            rows.append([device] + r)

    headers = [
        "Device",
        "Framework",
        "Model",
        "Input Size",
        "Mean (ms)",
        "Median (ms)",
        "Min (ms)",
        "Max (ms)",
    ]
    return _make_rst_table(rows, headers)


def _make_device_table(
    data: dict[str, dict[str, dict[str, dict[str, float]]]],
) -> str:
    rows = _get_device_rows(data)

    headers = [
        "Framework",
        "Model",
        "Input Size",
        "Mean (ms)",
        "Median (ms)",
        "Min (ms)",
        "Max (ms)",
    ]
    return _make_rst_table(rows, headers)


def main() -> None:
    """Generate RST tables of benchmarking results."""
    data_dir = Path(__file__).parent / "data"
    data: dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]] = {}
    for file in data_dir.iterdir():
        if file.is_dir():
            continue
        if file.suffix != ".json":
            continue

        name = file.stem
        with file.open("r") as f:
            file_data: dict[str, dict[str, dict[str, dict[str, float]]]] = json.load(f)
        data[name] = file_data

    rst_dir = Path(__file__).parent / "rst"
    rst_dir.mkdir(parents=True, exist_ok=True)

    # create the overall table
    print("Making overall table...")
    all_table = _make_all_table(data)
    all_table_path = rst_dir / "overview.rst"
    with all_table_path.open("w+") as f:
        f.write(all_table)

    # create each device table
    device_dir = rst_dir / "devices"
    device_dir.mkdir(parents=True, exist_ok=True)
    for device, device_data in data.items():
        print(f"Mkaing {device} table...")
        device_table = _make_device_table(device_data)
        device_table_path = device_dir / f"{device}.rst"
        with device_table_path.open("w+") as f:
            f.write(device_table)


if __name__ == "__main__":
    main()
