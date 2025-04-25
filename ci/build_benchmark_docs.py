#!/usr/bin/env python3
# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
import subprocess
from pathlib import Path
import json

# Global path definitions
ROOT_DIR = Path(__file__).parent.parent
BENCHMARK_DIR = ROOT_DIR / "benchmark"
PLOTS_DIR = BENCHMARK_DIR / "plots"
RST_DIR = ROOT_DIR / "docs" / "benchmark"
TABLES_DIR = BENCHMARK_DIR / "rst" / "devices"

SKIP_DEVICES = ["3090"]


def run_benchmark_scripts():
    """Run the benchmark scripts to generate tables and plots."""
    print("Generating benchmark tables...")
    subprocess.run(
        ["python3", str(BENCHMARK_DIR / "table.py"), "--overwrite"], check=True
    )

    print("Generating benchmark plots...")
    subprocess.run(
        ["python3", str(BENCHMARK_DIR / "plot.py")], check=True
    )


def generate_rst_docs():
    """Generate RST documentation from benchmark data."""
    run_benchmark_scripts()
    RST_DIR.mkdir(parents=True, exist_ok=True)

    # Load device info blurbs
    device_info_path = BENCHMARK_DIR / "device_info.json"
    if device_info_path.exists():
        with open(device_info_path, "r") as f:
            device_info = json.load(f)
    else:
        device_info = {}

    # Get all device table files
    all_device_files = list(TABLES_DIR.glob("*.rst"))
    device_files = []
    for df in all_device_files:
        for skip_device in SKIP_DEVICES:
            if skip_device in df.stem:
                break
        else:
            device_files.append(df)
    device_files.sort()

    # Split into Jetson and RTX/GTX
    jetson_devices = []
    rtx_devices = []
    for device_file in device_files:
        name = device_file.stem
        if ("Xavier" in name) or ("Orin" in name):
            jetson_devices.append(device_file)
        else:
            rtx_devices.append(device_file)

    # Generate main benchmark RST file
    with open(RST_DIR / "index.rst", "w") as f:
        f.write("""Benchmarking Results
====================

This section contains benchmarking results for various YOLO models using different frameworks on different devices.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   jetson
   rtx
""")

    # Generate Jetson and RTX/GTX summary files
    for group, group_name in [(jetson_devices, "jetson"), (rtx_devices, "rtx")]:
        with open(RST_DIR / f"{group_name}.rst", "w") as f:
            title = f"{'Jetson Devices' if group_name == 'jetson' else 'RTX/GTX Devices'}"
            f.write(f"""{title}
{'=' * len(title)}

.. toctree::
   :maxdepth: 1

""")
            for device_file in group:
                device_name = device_file.stem
                f.write(f"   {device_name.lower()}\n")
            f.write("\n")
            for device_file in group:
                device_name = device_file.stem
                blurb = device_info.get(device_name, "")
                if blurb:
                    f.write(f"**{device_name}**: {blurb}\n\n")
                else:
                    f.write(f"**{device_name}**\n\n")

    # Generate RST file for each device
    for device_file in device_files:
        device_name = device_file.stem
        rst_file = RST_DIR / f"{device_name.lower()}.rst"
        with open(rst_file, "w") as f:
            f.write(f"""{device_name}
{'=' * len(device_name)}

""")
            blurb = device_info.get(device_name, "")
            if blurb:
                f.write(f"{blurb}\n\n")
            f.write(f"This section contains benchmarking results for various YOLO models on the {device_name} platform.\n\n")
            f.write("""
Performance Plots
-----------------

The following plots show the performance comparison between different YOLO models and frameworks:

""")
            device_plots_dir = PLOTS_DIR / device_name
            if device_plots_dir.exists():
                for plot_file in sorted(device_plots_dir.glob("*.png")):
                    plot_name = plot_file.stem
                    f.write(f"""
{plot_name}
~~~~~~~~

.. image:: ../../benchmark/plots/{device_name}/{plot_name}.png
   :alt: {plot_name} performance plot
   :align: center

""")
            f.write("""
Performance Table
-----------------

The following table shows detailed performance metrics for all tested models:

""")
            with open(device_file, "r") as table_file:
                f.write(table_file.read())


if __name__ == "__main__":
    generate_rst_docs()
