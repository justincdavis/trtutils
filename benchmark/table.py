# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate RST tables of benchmarking results.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing tables.",
    )
    parser.add_argument(
        "--skip-devices",
        type=str,
        default="",
        help="Comma-separated list of device names to skip.",
    )
    args = parser.parse_args()
    
    skip_devices = set()
    if args.skip_devices:
        skip_devices = {d.strip() for d in args.skip_devices.split(",") if d.strip()}
    
    data_dir = Path(__file__).parent / "data" / "models"
    data: dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]] = {}
    for file in data_dir.iterdir():
        if file.is_dir():
            continue
        if file.suffix != ".json":
            continue

        name = file.stem
        if name in skip_devices:
            print(f"Skipping device: {name}")
            continue
            
        try:
            with file.open("r") as f:
                file_data: dict[str, dict[str, dict[str, dict[str, float]]]] = json.load(f)
            data[name] = file_data
        except Exception as e:
            print(f"Warning: Failed to load data for {name}: {e}, skipping...")
            continue

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
        print(f"Making {device} table...")
        device_table = _make_device_table(device_data)
        device_table_path = device_dir / f"{device}.rst"
        with device_table_path.open("w+") as f:
            f.write(device_table)


if __name__ == "__main__":
    import sys
    try:
        main()
    except Exception as e:
        print(f"Error in table.py: {e}", file=sys.stderr)
        sys.exit(1)
