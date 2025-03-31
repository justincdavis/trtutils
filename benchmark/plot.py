# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Generate plots of benchmarking results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def get_data() -> list[tuple[str, dict[str, dict[str, dict[str, dict[str, float]]]]]]:
    data_dir = Path(__file__).parent / "data"

    device_data: list[tuple[str, dict[str, dict[str, dict[str, dict[str, float]]]]]] = []

    for file in data_dir.iterdir():
        device_name = file.stem
        with file.open("r") as f:
            data = json.load(f)
        
        device_data.append((device_name, data))
    
    return device_data


def plot_device(name: str, data: dict[str, dict[str, dict[str, dict[str, float]]]], *, overwrite: bool) -> None:
    plot_dir = Path(__file__).parent / "plots" / name
    plot_dir.mkdir(parents=True, exist_ok=True)

    # get all frameworks
    frameworks: list[str] = sorted(list(data.keys()), reverse=True)
    # get all models
    model_set = set()
    for f in frameworks:
        for m in data[f]:
            model_set.add(m)
    models: list[str] = list(model_set)
    models = sorted(models)
    # get all image sizes, should be same for all models
    image_sizes = sorted([int(imgsz) for imgsz in data[frameworks[0]][models[0]]])

    # print(frameworks)
    # print(models)
    # print(image_sizes)

    for model in models:
        plot_path = plot_dir / f"{model}.png"
        if plot_path.exists() and not overwrite:
            continue
        
        print(f"Plotting: {model}")

        # unfold the model name so we have framework -> metrics
        mdata: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for f in frameworks:
            model_data = data[f].get(model)
            if model_data is None:
                continue

            # if this model is present then add the data
            for imgsz in data[f][model]:
                mdata[f].append((int(imgsz), data[f][model][imgsz]["mean"]))

        # image_sizes = sorted({size for framework in mdata for (size, _) in mdata[framework]})
        sub_frameworks = list(mdata.keys())
        n_groups = len(image_sizes)
        x = np.arange(n_groups)
        n_frameworks = len(sub_frameworks)
        bar_width = 0.8 / n_frameworks
        plt.figure(figsize=(8, 5))

        for i, framework in enumerate(sub_frameworks):
            latencies = [latency for _, latency in mdata[framework]]
            plt.bar(x + i * bar_width, latencies, width=bar_width, label=framework)

        plt.xlabel("Input Size")
        plt.ylabel("Latency (ms)")
        plt.title(f"Input Size and Latency Comparision for {model}")
        plt.xticks(x + bar_width * (n_frameworks - 1) / 2, image_sizes)
        plt.legend(title="Framework")
        plt.tight_layout()
        plt.savefig(plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate plots for each device based on benchmark results.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing plots.",
    )
    args = parser.parse_args()

    # parse all the data
    all_data = get_data()
    for name, data in all_data:
        plot_device(name, data, overwrite=args.overwrite)
