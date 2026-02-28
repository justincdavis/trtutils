# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: T201
"""Model latency bar charts and Pareto frontier plots."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from utils.config import DATA_DIR, IMAGE_SIZES, MODEL_FAMILIES, MODEL_FRAMEWORKS

_PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
_INFO_DIR = Path(__file__).resolve().parent.parent / "info"

COLORS = {fm: plt.cm.tab10(idx) for idx, fm in enumerate(MODEL_FRAMEWORKS)}
FAMILY_COLORS = {fm: plt.cm.tab10(idx) for idx, fm in enumerate(MODEL_FAMILIES)}
EPSILON = 1e-9


def load_all_device_data() -> (
    list[tuple[str, dict[str, dict[str, dict[str, dict[str, float]]]]]]
):
    """Load benchmark data for all devices."""
    data_dir = DATA_DIR / "models"
    device_data: list[
        tuple[str, dict[str, dict[str, dict[str, dict[str, float]]]]]
    ] = []

    for file in data_dir.iterdir():
        if file.is_dir() or file.suffix != ".json":
            continue
        try:
            with file.open("r") as f:
                data = json.load(f)
            device_data.append((file.stem, data))
        except Exception as e:
            print(f"Warning: Failed to load data for {file.stem}: {e}, skipping...")

    return device_data


def load_model_info() -> dict[str, dict[str, float]]:
    """Load model metadata (params, mAP, etc.) from info files."""
    info_dir = _INFO_DIR / "model_info"
    overall_data: dict[str, dict[str, float]] = {}

    for file in info_dir.iterdir():
        if not file.is_file():
            continue
        with file.open("r") as f:
            data: dict[str, dict[str, float]] = json.load(f)
        for model, info in data.items():
            overall_data[model] = info

    return overall_data


def get_model_family(model_name: str) -> str:
    """Map a model name to its family (e.g. 'yolov10n' -> 'yolov10')."""
    for family in MODEL_FAMILIES:
        if model_name.startswith(family):
            return family
    return "unknown"


def compute_pareto_frontier(points: list[tuple[float, float]]) -> list[bool]:
    """Compute Pareto frontier mask for (cost, accuracy) pairs.

    Returns a boolean list where True indicates a Pareto-optimal point.
    """
    n = len(points)
    if n == 0:
        return []

    indexed_points = [(*pt, idx) for idx, pt in enumerate(points)]
    indexed_points.sort(key=lambda item: (item[0], -item[1]))

    pareto_candidates: list[tuple[float, float, int]] = []
    best_accuracy = float("-inf")
    for cost, accuracy, idx in indexed_points:
        if accuracy > best_accuracy + EPSILON:
            pareto_candidates.append((cost, accuracy, idx))
            best_accuracy = accuracy

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
    """Generate per-model latency bar charts for a device."""
    plot_dir = _PLOT_DIR / name
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Collect all models across frameworks
    model_set: set[str] = set()
    for f in MODEL_FRAMEWORKS:
        framework_data = data.get(f)
        if framework_data is not None:
            model_set.update(framework_data)
    models = sorted(model_set)

    plt.style.use("seaborn-v0_8")

    fontsize = 12
    title_fontsize = fontsize + 4
    label_fontsize = fontsize + 1
    tick_fontsize = fontsize + 1
    annotation_fontsize = fontsize
    legend_fontsize = fontsize
    subtitle_fontsize = fontsize

    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            label = f"{height:.3g}"[:4]
            ax.annotate(
                label,
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                rotation=90,
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=annotation_fontsize,
            )

    print(f"Plotting - {name}")
    for model in models:
        plot_path = plot_dir / f"{model}.png"
        if plot_path.exists() and not overwrite:
            continue

        print(f"\t{model}")

        # Gather framework -> [(imgsz, mean_latency)] data
        mdata: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for f in MODEL_FRAMEWORKS:
            model_data = (data.get(f) or {}).get(model)
            if model_data is None:
                continue
            for imgsz_str, metrics in model_data.items():
                mdata[f].append((int(imgsz_str), metrics["mean"]))

        sub_frameworks = list(mdata.keys())
        n_groups = len(IMAGE_SIZES)
        x = np.arange(n_groups)
        n_fw = len(sub_frameworks)
        bar_width = 0.9 / n_fw
        _, ax = plt.subplots(figsize=(11, 6))

        max_latency = 0.0
        for i, framework in enumerate(sub_frameworks):
            latencies = [latency for _, latency in mdata[framework]]
            max_latency = max(max_latency, max(latencies) if latencies else 0)

            rects = ax.bar(
                x + i * bar_width, latencies,
                width=bar_width, label=framework,
                color=COLORS[framework],
            )
            autolabel(rects, ax)

        ax.set_ylim(0, max_latency * 1.15)
        plt.xlabel("Input Size", fontsize=label_fontsize)
        plt.ylabel("Latency (ms)", fontsize=label_fontsize)
        plt.suptitle(
            f"{name} - Input Size and Latency Comparison for {model}",
            y=0.95, fontsize=title_fontsize,
        )
        plt.title("Batch 1, end-to-end latency", fontsize=subtitle_fontsize)
        plt.xticks(
            x + bar_width * (n_fw - 1) / 2,
            IMAGE_SIZES, fontsize=tick_fontsize,
        )
        plt.yticks(fontsize=tick_fontsize)
        plt.legend(
            title="Framework",
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
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
    """Generate Pareto frontier plot (accuracy vs latency)."""
    plot_dir = _PLOT_DIR / name
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_path = plot_dir / "pareto.png"
    if plot_path.exists() and not overwrite:
        return

    print(f"\t{framework}")

    framework_data = data.get(framework)
    if framework_data is None:
        print(f"\t  No data for framework: {framework}")
        return

    # Collect data points: (cost, accuracy, model_name, family)
    points_data: list[tuple[float, float, str, str]] = []
    for model_name, model_benchmarks in framework_data.items():
        if model_name not in model_info:
            continue
        accuracy = model_info[model_name].get("coco_map_50")
        if accuracy is None:
            continue
        family = get_model_family(model_name)
        if family not in MODEL_FAMILIES:
            continue

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

    pareto_mask = compute_pareto_frontier(
        [(cost, acc) for cost, acc, _, _ in points_data],
    )

    plt.style.use("seaborn-v0_8")
    _, ax = plt.subplots(figsize=(12, 8))

    fontsize = 16
    title_fontsize = fontsize + 6
    label_fontsize = fontsize + 4
    tick_fontsize = fontsize + 2
    legend_fontsize = fontsize + 2

    for family in MODEL_FAMILIES:
        family_points = [
            (cost, acc, model, is_pareto)
            for (cost, acc, model, fam), is_pareto in zip(points_data, pareto_mask)
            if fam == family
        ]
        if not family_points:
            continue

        optimal = [(c, a, m) for c, a, m, p in family_points if p]
        non_optimal = [(c, a, m) for c, a, m, p in family_points if not p]
        color = FAMILY_COLORS[family]

        # Connecting lines
        all_pts = sorted(
            [(c, a, m) for c, a, m, _ in family_points], key=lambda x: x[0],
        )
        if len(all_pts) > 1:
            costs, accs, _ = zip(*all_pts)
            ax.plot(
                costs, accs, color=color,
                alpha=0.25, linewidth=2.0, linestyle="-", zorder=1,
            )

        # Non-optimal points
        if non_optimal:
            costs, accs, _ = zip(*non_optimal)
            ax.scatter(
                costs, accs, color=color, alpha=0.4, s=200,
                marker="o", edgecolors="none", zorder=2,
            )
            for cost, acc, model in non_optimal:
                ax.annotate(
                    model, xy=(cost, acc), xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=fontsize - 2, alpha=0.5,
                )

        # Optimal points
        if optimal:
            costs, accs, _ = zip(*optimal)
            ax.scatter(
                costs, accs, color=color, alpha=1.0, s=300,
                marker="o", label=family,
                edgecolors="black", linewidths=2.5, zorder=3,
            )
            for cost, acc, model in optimal:
                ax.annotate(
                    model, xy=(cost, acc), xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=fontsize, fontweight="bold",
                )

    # Pareto frontier line
    pareto_pts = sorted(
        [
            (cost, acc)
            for (cost, acc, _, _), is_pareto in zip(points_data, pareto_mask)
            if is_pareto
        ],
        key=lambda x: x[0],
    )
    if pareto_pts:
        costs, accs = zip(*pareto_pts)
        ax.plot(
            costs, accs, "k--",
            alpha=0.4, linewidth=2.5, label="Pareto Frontier",
        )

    ax.set_xlabel("Latency (ms)", fontsize=label_fontsize, fontweight="bold")
    ax.set_ylabel("COCO mAP@50", fontsize=label_fontsize, fontweight="bold")
    ax.set_title(
        f"{name} - Accuracy vs. Latency - {framework}",
        fontsize=title_fontsize, fontweight="bold", pad=20,
    )
    ax.tick_params(axis="both", labelsize=tick_fontsize, width=1.5, length=6)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=1.2)
    ax.legend(
        fontsize=legend_fontsize, loc="best",
        framealpha=0.9, edgecolor="black", fancybox=True,
    )

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
