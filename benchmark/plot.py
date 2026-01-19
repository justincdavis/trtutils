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
import numpy as np
import matplotlib.pyplot as plt

from run import FRAMEWORKS

matplotlib.use("Agg")

IMAGE_SIZES = [160, 320, 480, 640, 800, 960, 1120, 1280]
COLORS = {
    fm: plt.cm.tab10(idx) for idx, fm in enumerate(FRAMEWORKS)
}
MODEL_FAMILIES = [
    "yolov13",
    "yolov12",
    "yolov11",
    "yolov10",
    "yolov9",
    "yolov8",
    "yolov7",
    "yolox",
]
FAMILY_COLORS = {
    fm: plt.cm.tab10(idx) for idx, fm in enumerate(MODEL_FAMILIES)
}

EPSILON = 1e-9


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


def get_model_info() -> dict[str, dict[str, float]]:
    info_dir = Path(__file__).parent / "info" / "model_info"
    overall_data: dict[str, dict[str, float]] = {}

    # for each file in the directory
    for file in info_dir.iterdir():
        if not file.is_file():
            continue

        # load the data
        with file.open("r") as f:
            data: dict[str, dict[str, float]] = json.load(f)

        # add each entry to the overall data
        for model, info in data.items():
            overall_data[model] = info

    return overall_data


def get_model_family(model_name: str) -> str:
    for family in MODEL_FAMILIES:
        if model_name.startswith(family):
            return family
    return "unknown"


def compute_pareto_frontier(points: list[tuple[float, float]]) -> list[bool]:
    n = len(points)
    if n == 0:
        return []

    indexed_points = [(*pt, idx) for idx, pt in enumerate(points)]
    indexed_points.sort(key=lambda item: (item[0], -item[1]))

    pareto_candidates: list[tuple[float, float, int]] = []
    best_accuracy_so_far = float("-inf")
    for cost, accuracy, idx in indexed_points:
        if accuracy > best_accuracy_so_far + EPSILON:
            pareto_candidates.append((cost, accuracy, idx))
            best_accuracy_so_far = accuracy

    if not pareto_candidates:
        return [False] * n

    frontier: list[tuple[float, float, int]] = []
    for point in pareto_candidates:
        while len(frontier) >= 2:
            c1, a1, _ = frontier[-2]
            c2, a2, _ = frontier[-1]
            c3, a3, _ = point
            slope12 = (a2 - a1) * (c3 - c2)
            slope23 = (a3 - a2) * (c2 - c1)
            if slope12 + EPSILON < slope23:
                frontier.pop()
            else:
                break
        frontier.append(point)

    mask = [False] * n
    for _, _, idx in frontier:
        mask[idx] = True

    return mask


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


def plot_pareto(
    name: str,
    data: dict[str, dict[str, dict[str, dict[str, float]]]],
    model_info: dict[str, dict[str, float]],
    framework: str,
    *,
    overwrite: bool,
) -> None:
    plot_dir = Path(__file__).parent / "plots" / name
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_path = plot_dir / "pareto.png"
    if plot_path.exists() and not overwrite:
        return

    print(f"\t{framework}")

    # Extract framework data
    framework_data = data.get(framework)
    if framework_data is None:
        print(f"\t  No data for framework: {framework}")
        return

    # Collect data points
    points_data = []  # (cost, accuracy, model_name, family)

    for model_name, model_benchmarks in framework_data.items():
        if model_name not in model_info:
            continue

        accuracy = model_info[model_name].get("coco_map_50")
        if accuracy is None:
            continue

        family = get_model_family(model_name)
        if family not in MODEL_FAMILIES:
            continue

        # Grab the median latency of the image size
        # that is presnet in the model_info entry
        imgsz = str(model_info[model_name]["imgsz"])
        entry = model_benchmarks.get(imgsz)
        if entry is None:
            continue
        cost = entry.get("median")
        if cost is not None:
            points_data.append((cost, accuracy, model_name, family))

    if not points_data:
        print(f"\t  No valid data points for {framework}")
        return

    # Determine Pareto optimal points
    points_only = [(cost, acc) for cost, acc, _, _ in points_data]
    pareto_mask = compute_pareto_frontier(points_only)

    # Create plot
    plt.style.use("seaborn-v0_8")
    _, ax = plt.subplots(figsize=(12, 8))

    # Define font sizes
    fontsize = 16
    title_fontsize = fontsize + 6
    label_fontsize = fontsize + 4
    tick_fontsize = fontsize + 2
    legend_fontsize = fontsize + 2

    # Plot each family separately
    for family in MODEL_FAMILIES:
        family_points = [
            (cost, acc, model, is_pareto)
            for (cost, acc, model, fam), is_pareto in zip(points_data, pareto_mask)
            if fam == family
        ]

        if not family_points:
            continue

        # Separate Pareto optimal and non-optimal points
        optimal_points = [(c, a, m) for c, a, m, p in family_points if p]
        non_optimal_points = [(c, a, m) for c, a, m, p in family_points if not p]

        color = FAMILY_COLORS[family]

        # Draw connecting lines for all points in the family
        all_family_points = [(c, a, m) for c, a, m, p in family_points]
        if len(all_family_points) > 1:
            # Sort by cost (latency) for logical connection order
            all_family_points_sorted = sorted(all_family_points, key=lambda x: x[0])
            costs, accs, _ = zip(*all_family_points_sorted)
            ax.plot(
                costs,
                accs,
                color=color,
                alpha=0.25,
                linewidth=2.0,
                linestyle="-",
                zorder=1,
            )

        # Plot non-optimal points with reduced alpha
        if non_optimal_points:
            costs, accs, models = zip(*non_optimal_points)
            ax.scatter(
                costs,
                accs,
                color=color,
                alpha=0.4,
                s=200,
                marker="o",
                edgecolors="none",
                zorder=2,
            )
            # Add labels for non-optimal points
            for cost, acc, model in non_optimal_points:
                ax.annotate(
                    model,
                    xy=(cost, acc),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=fontsize - 2,
                    alpha=0.5,
                )

        # Plot optimal points with full alpha
        if optimal_points:
            costs, accs, models = zip(*optimal_points)
            ax.scatter(
                costs,
                accs,
                color=color,
                alpha=1.0,
                s=300,
                marker="o",
                label=family,
                edgecolors="black",
                linewidths=2.5,
                zorder=3,
            )
            # Add labels for optimal points
            for cost, acc, model in optimal_points:
                ax.annotate(
                    model,
                    xy=(cost, acc),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=fontsize,
                    fontweight="bold",
                )

    # Draw Pareto frontier line
    pareto_points = [
        (cost, acc)
        for (cost, acc, _, _), is_pareto in zip(points_data, pareto_mask)
        if is_pareto
    ]
    if pareto_points:
        # Sort by cost
        pareto_points_sorted = sorted(pareto_points, key=lambda x: x[0])
        costs, accs = zip(*pareto_points_sorted)
        ax.plot(
            costs,
            accs,
            "k--",
            alpha=0.4,
            linewidth=2.5,
            label="Pareto Frontier",
        )

    ax.set_xlabel("Latency (ms)", fontsize=label_fontsize, fontweight="bold")
    ax.set_ylabel("COCO mAP@50", fontsize=label_fontsize, fontweight="bold")
    ax.set_title(
        f"{name} - Accuracy vs. Latency - {framework}",
        fontsize=title_fontsize,
        fontweight="bold",
        pad=20,
    )
    ax.tick_params(axis="both", labelsize=tick_fontsize, width=1.5, length=6)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=1.2)
    ax.legend(fontsize=legend_fontsize, loc="best", framealpha=0.9, edgecolor="black", fancybox=True)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Generate plots for each device based on benchmark results."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="The device to make the plots for. If not specified, all devices will be used.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing plots.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="The model to generate plots for. If not specified, all models will be used.",
    )
    parser.add_argument(
        "--latency",
        action="store_true",
        help="Generate latency plots.",
    )
    parser.add_argument(
        "--pareto",
        action="store_true",
        help="Generate Pareto frontier plots.",
    )
    parser.add_argument(
        "--framework",
        type=str,
        default="trtutils(trt)",
        help="Framework to use for pareto optimal plot (default: trtutils(trt)).",
    )
    args = parser.parse_args()

    # parse all the data
    all_data = get_data()
    
    # Load model info for pareto optimal plots
    model_info = get_model_info()

    for name, data in all_data:
        if args.device is None or name == args.device:
            if args.latency:
                plot_device(name, data, overwrite=args.overwrite)
            if args.pareto:
                plot_pareto(
                    name, data, model_info, args.framework, overwrite=args.overwrite
                )
