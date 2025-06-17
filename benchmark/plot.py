# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Generate plots of benchmarking results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

from run import FRAMEWORKS

IMAGE_SIZES = [160, 320, 480, 640, 800, 960, 1120, 1280]
COLORS = {fm: plt.cm.tab10(idx) for idx, fm in enumerate(FRAMEWORKS)}


def get_data() -> list[tuple[str, dict[str, dict[str, dict[str, dict[str, float]]]]]]:
    data_dir = Path(__file__).parent / "data"

    device_data: list[
        tuple[str, dict[str, dict[str, dict[str, dict[str, float]]]]]
    ] = []

    for file in data_dir.iterdir():
        device_name = file.stem
        with file.open("r") as f:
            data = json.load(f)

        device_data.append((device_name, data))

    return device_data


def plot_device(
    name: str,
    data: dict[str, dict[str, dict[str, dict[str, float]]]],
    *,
    overwrite: bool,
) -> None:
    plot_dir = Path(__file__).parent / "plots" / name
    plot_dir.mkdir(parents=True, exist_ok=True)

    # # get all frameworks
    # frameworks: list[str] = sorted(list(data.keys()), reverse=True)
    # get all models
    model_set = set()
    for f in FRAMEWORKS:
        framework_data = data.get(f)
        if framework_data is None:
            continue
        for m in framework_data:
            model_set.add(m)
    models: list[str] = list(model_set)
    models = sorted(models)

    # print(frameworks)
    # print(models)
    # print(image_sizes)

    plt.style.use("seaborn-v0_8")

    print(f"Plotting - {name}")
    for model in models:
        plot_path = plot_dir / f"{model}.png"
        if plot_path.exists() and not overwrite:
            continue

        print(f"\t{model}")

        # unfold the model name so we have framework -> metrics
        mdata: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for f in FRAMEWORKS:
            framework_data = data.get(f)
            if framework_data is None:
                continue
            model_data = framework_data.get(model)
            if model_data is None:
                continue

            # if this model is present then add the data
            for imgsz in data[f][model]:
                mdata[f].append((int(imgsz), data[f][model][imgsz]["mean"]))

        # image_sizes = sorted({size for framework in mdata for (size, _) in mdata[framework]})
        sub_frameworks = list(mdata.keys())
        n_groups = len(IMAGE_SIZES)
        x = np.arange(n_groups)
        n_frameworks = len(sub_frameworks)
        bar_width = 0.9 / n_frameworks
        _, ax = plt.subplots(figsize=(11, 6))

        # Define font sizes based on a base font size
        fontsize = 12
        title_fontsize = fontsize + 4
        label_fontsize = fontsize + 1
        tick_fontsize = fontsize + 1
        annotation_fontsize = fontsize
        legend_fontsize = fontsize
        subtitle_fontsize = fontsize

        def autolabel(rects, ax):
            def format_float(value):
                formatted = f"{value:.3g}"
                return formatted[:4]

            for rect in rects:
                height = rect.get_height()
                ax.annotate(
                    format_float(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    rotation=90,
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=annotation_fontsize,
                )

        # Find the maximum latency value to set y-axis limit
        max_latency = 0
        for framework in sub_frameworks:
            latencies = [latency for _, latency in mdata[framework]]
            max_latency = max(max_latency, max(latencies) if latencies else 0)

        for i, framework in enumerate(sub_frameworks):
            latencies = [latency for _, latency in mdata[framework]]
            rects = ax.bar(
                x + i * bar_width,
                latencies,
                width=bar_width,
                label=framework,
                color=COLORS[framework],
            )
            autolabel(rects, ax)

        # Set y-axis limit with 10% padding to ensure all values and annotations fit
        ax.set_ylim(0, max_latency * 1.15)

        plt.xlabel("Input Size", fontsize=label_fontsize)
        plt.ylabel("Latency (ms)", fontsize=label_fontsize)
        plt.suptitle(
            f"{name} - Input Size and Latency Comparision for {model}",
            y=0.95,
            fontsize=title_fontsize,
        )
        plt.title("Batch 1, end-to-end latency", fontsize=subtitle_fontsize)
        plt.xticks(
            x + bar_width * (n_frameworks - 1) / 2, IMAGE_SIZES, fontsize=tick_fontsize
        )
        plt.yticks(fontsize=tick_fontsize)
        plt.legend(
            title="Framework", fontsize=legend_fontsize, title_fontsize=legend_fontsize
        )
        plt.tight_layout()
        plt.savefig(plot_path)

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Generate plots for each device based on benchmark results."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="The device to make the plots for, optional.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing plots.",
    )
    args = parser.parse_args()

    # parse all the data
    all_data = get_data()
    for name, data in all_data:
        if args.device is None or name == args.device:
            plot_device(name, data, overwrite=args.overwrite)
