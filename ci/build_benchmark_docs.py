#!/usr/bin/env python3
# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
import subprocess
from pathlib import Path

# Global path definitions
ROOT_DIR = Path(__file__).parent.parent
BENCHMARK_DIR = ROOT_DIR / "benchmark"
PLOTS_DIR = BENCHMARK_DIR / "plots"
RST_DIR = ROOT_DIR / "docs" / "benchmark"
TABLES_DIR = BENCHMARK_DIR / "rst" / "devices"

def run_benchmark_scripts():
    """Run the benchmark scripts to generate tables and plots."""
    print("Generating benchmark tables...")
    subprocess.run(["python", str(BENCHMARK_DIR / "table.py")], check=True)
    
    print("Generating benchmark plots...")
    subprocess.run(["python", str(BENCHMARK_DIR / "plot.py")], check=True)

def generate_rst_docs():
    """Generate RST documentation from benchmark data."""
    # Run the benchmark scripts to ensure everything is up to date
    run_benchmark_scripts()
    
    # Create RST directory if it doesn't exist
    RST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all device table files
    device_files = list(TABLES_DIR.glob("*.rst"))
    
    # Generate main benchmark RST file
    with open(RST_DIR / "index.rst", "w") as f:
        f.write("""Benchmarking Results
===================

This section contains benchmarking results for various YOLO models using different frameworks on different devices.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

""")
        # Add all devices to the table of contents
        for device_file in device_files:
            device_name = device_file.stem.lower()
            f.write(f"   {device_name}\n")
    
    # Generate RST file for each device
    for device_file in device_files:
        device_name = device_file.stem
        rst_file = RST_DIR / f"{device_name.lower()}.rst"
        
        with open(rst_file, "w") as f:
            f.write(f"""{device_name}
{'=' * len(device_name)}

This section contains benchmarking results for various YOLO models on the {device_name} platform.

Performance Plots
----------------

The following plots show the performance comparison between different YOLO models and frameworks:

""")
            
            # Add plots if they exist for this device
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
            
            # Add performance table
            f.write("""
Performance Table
----------------

The following table shows detailed performance metrics for all tested models:

""")
            
            # Include the pre-generated RST table
            with open(device_file, "r") as table_file:
                f.write(table_file.read())

if __name__ == "__main__":
    generate_rst_docs()
