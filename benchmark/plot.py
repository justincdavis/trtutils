# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
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
    data_dir = Path(__file__).parent / "data" / "models"

    device_data: list[
        tuple[str, dict[str, dict[str, dict[str, dict[str, float]]]]]
    ] = []

    for file in data_dir.iterdir():
        if file.is_dir() or file.suffix != ".json":
            continue
            
        device_name = file.stem
        try:
            with file.open("r") as f:
                data = json.load(f)
            device_data.append((device_name, data))
        except Exception as e:
            print(f"Warning: Failed to load data for {device_name}: {e}, skipping...")
            continue

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


# Seven configs for optimization bar chart (plan order): baseline then cuda-only configs
OPTIMIZATION_BAR_CONFIGS: list[tuple[str, str, bool, bool, bool]] = [
    ("CPU (baseline)", "cpu", False, False, False),
    ("GPU preproc", "cuda", False, False, False),
    ("+ Pagelocked", "cuda", False, True, False),
    ("+ Unified", "cuda", False, False, True),
    ("CUDA graph", "cuda", True, False, False),
    ("CUDA graph + Pagelocked", "cuda", True, True, False),
    ("CUDA graph + Unified", "cuda", True, False, True),
]


def _find_result(
    results: list[dict],
    preprocessor: str,
    cuda_graph: bool,
    pagelocked_mem: bool,
    unified_mem: bool,
) -> dict | None:
    """Return the first result dict matching the four flags, or None."""
    for r in results:
        if (
            r.get("preprocessor") == preprocessor
            and r.get("cuda_graph") is cuda_graph
            and r.get("pagelocked_mem") is pagelocked_mem
            and r.get("unified_mem") is unified_mem
        ):
            return r
    return None


def load_optimization_data(device: str) -> list[float] | None:
    """
    Load data/optimizations/{device}.json and return speedups for the seven
    bar configs (baseline = 1.0). Missing configs yield 0.0 for that bar.
    Returns None if the file is missing or invalid.
    """
    opt_dir = Path(__file__).parent / "data" / "optimizations"
    path = opt_dir / f"{device}.json"
    if not path.exists():
        return None
    try:
        with path.open("r") as f:
            payload = json.load(f)
    except Exception:
        return None
    # Accept either a list of results or {"results": [...]}
    if isinstance(payload, list):
        results = payload
    elif isinstance(payload, dict) and "results" in payload:
        results = payload["results"]
    else:
        return None
    baseline_ms: float | None = None
    for _label, prep, cg, pl, um in OPTIMIZATION_BAR_CONFIGS:
        r = _find_result(results, prep, cg, pl, um)
        if r is not None and prep == "cpu":
            baseline_ms = r.get("mean_ms")
            break
    if baseline_ms is None or baseline_ms <= 0:
        return None
    speedups: list[float] = []
    for _label, prep, cg, pl, um in OPTIMIZATION_BAR_CONFIGS:
        r = _find_result(results, prep, cg, pl, um)
        if r is not None:
            mean_ms = r.get("mean_ms")
            if mean_ms is not None and mean_ms > 0:
                speedups.append(baseline_ms / mean_ms)
            else:
                speedups.append(0.0)
        else:
            speedups.append(0.0)
    return speedups


def plot_optimizations(device: str, *, overwrite: bool) -> None:
    """
    Plot per-device optimization bar chart (speedup vs CPU baseline) to
    plots/{device}/optimizations.png.
    """
    speedups = load_optimization_data(device)
    if speedups is None:
        print(f"Warning: No optimization data for {device}, skipping...")
        return
    labels = [c[0] for c in OPTIMIZATION_BAR_CONFIGS]
    plot_dir = Path(__file__).parent / "plots" / device
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "optimizations.png"
    if plot_path.exists() and not overwrite:
        return
    plt.style.use("seaborn-v0_8")
    _, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(labels))
    ax.bar(x, speedups, width=0.7)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Speedup (vs CPU baseline)")
    ax.set_title(f"{device} - Optimization speedup (CUDA preprocessor)")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved {plot_path}")


BATCH_FRAMEWORKS = [
    "trtutils",
    "trtutils(graph)",
    "ultralytics(trt)",
    "ultralytics(torch)",
]
BATCH_COLORS = {
    fw: plt.cm.tab10(idx) for idx, fw in enumerate(BATCH_FRAMEWORKS)
}


def load_batch_data(
    device: str,
) -> dict[str, dict[str, dict[str, dict[str, float]]]] | None:
    """Load batch benchmark data for a device."""
    batch_dir = Path(__file__).parent / "data" / "batch"
    path = batch_dir / f"{device}.json"
    if not path.exists():
        return None
    with path.open("r") as f:
        return json.load(f)


def plot_batch_throughput(
    device: str,
    data: dict[str, dict[str, dict[str, dict[str, float]]]],
    *,
    overwrite: bool,
) -> None:
    """Line plot: batch size vs images/sec per framework."""
    plot_dir = Path(__file__).parent / "plots" / device
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "batch_throughput.png"
    if plot_path.exists() and not overwrite:
        return

    plt.style.use("seaborn-v0_8")
    _, ax = plt.subplots(figsize=(10, 6))

    fontsize = 14

    for fw in BATCH_FRAMEWORKS:
        fw_data = data.get(fw, {})
        # Collect all models' data (typically one model per batch run)
        for model_name, model_data in fw_data.items():
            batch_sizes = sorted(int(k) for k in model_data)
            throughputs = [model_data[str(bs)]["throughput"] for bs in batch_sizes]

            ax.plot(
                batch_sizes,
                throughputs,
                marker="o",
                linewidth=2,
                markersize=8,
                label=fw,
                color=BATCH_COLORS[fw],
            )

            # Annotate each point
            for bs, tp in zip(batch_sizes, throughputs):
                ax.annotate(
                    f"{tp:.0f}",
                    xy=(bs, tp),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=fontsize - 3,
                )

    ax.set_xlabel("Batch Size", fontsize=fontsize)
    ax.set_ylabel("Throughput (images/sec)", fontsize=fontsize)
    ax.set_title(
        f"{device} - Batch Size vs Throughput", fontsize=fontsize + 4, fontweight="bold",
    )
    ax.set_xscale("log", base=2)
    ax.tick_params(axis="both", labelsize=fontsize)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=fontsize - 1, loc="best")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {plot_path}")


def plot_batch_latency(
    device: str,
    data: dict[str, dict[str, dict[str, dict[str, float]]]],
    *,
    overwrite: bool,
) -> None:
    """Line plot: batch size vs latency (ms) per framework."""
    plot_dir = Path(__file__).parent / "plots" / device
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "batch_latency.png"
    if plot_path.exists() and not overwrite:
        return

    plt.style.use("seaborn-v0_8")
    _, ax = plt.subplots(figsize=(10, 6))

    fontsize = 14

    for fw in BATCH_FRAMEWORKS:
        fw_data = data.get(fw, {})
        for model_name, model_data in fw_data.items():
            batch_sizes = sorted(int(k) for k in model_data)
            latencies = [model_data[str(bs)]["mean"] for bs in batch_sizes]

            ax.plot(
                batch_sizes,
                latencies,
                marker="o",
                linewidth=2,
                markersize=8,
                label=fw,
                color=BATCH_COLORS[fw],
            )

            for bs, lat in zip(batch_sizes, latencies):
                ax.annotate(
                    f"{lat:.1f}",
                    xy=(bs, lat),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=fontsize - 3,
                )

    ax.set_xlabel("Batch Size", fontsize=fontsize)
    ax.set_ylabel("Latency (ms)", fontsize=fontsize)
    ax.set_title(
        f"{device} - Batch Size vs Latency", fontsize=fontsize + 4, fontweight="bold",
    )
    ax.set_xscale("log", base=2)
    ax.tick_params(axis="both", labelsize=fontsize)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=fontsize - 1, loc="best")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {plot_path}")


def plot_batch_scaling(
    device: str,
    data: dict[str, dict[str, dict[str, dict[str, float]]]],
    *,
    overwrite: bool,
) -> None:
    """Line plot: batch size vs scaling efficiency per framework.

    efficiency(N) = throughput_at_N / (N * throughput_at_1)
    1.0 = perfect linear scaling.
    """
    plot_dir = Path(__file__).parent / "plots" / device
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "batch_scaling.png"
    if plot_path.exists() and not overwrite:
        return

    plt.style.use("seaborn-v0_8")
    _, ax = plt.subplots(figsize=(10, 6))

    fontsize = 14

    for fw in BATCH_FRAMEWORKS:
        fw_data = data.get(fw, {})
        for model_name, model_data in fw_data.items():
            batch_sizes = sorted(int(k) for k in model_data)
            if not batch_sizes or "1" not in model_data:
                continue

            tp_at_1 = model_data["1"]["throughput"]
            if tp_at_1 <= 0:
                continue

            efficiencies = []
            for bs in batch_sizes:
                tp = model_data[str(bs)]["throughput"]
                efficiencies.append(tp / (bs * tp_at_1))

            ax.plot(
                batch_sizes,
                efficiencies,
                marker="o",
                linewidth=2,
                markersize=8,
                label=fw,
                color=BATCH_COLORS[fw],
            )

            for bs, eff in zip(batch_sizes, efficiencies):
                ax.annotate(
                    f"{eff:.2f}",
                    xy=(bs, eff),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=fontsize - 3,
                )

    # Draw perfect scaling reference line
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, label="Perfect scaling")

    ax.set_xlabel("Batch Size", fontsize=fontsize)
    ax.set_ylabel("Scaling Efficiency", fontsize=fontsize)
    ax.set_title(
        f"{device} - Batch Scaling Efficiency", fontsize=fontsize + 4, fontweight="bold",
    )
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1.3)
    ax.tick_params(axis="both", labelsize=fontsize)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=fontsize - 1, loc="best")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {plot_path}")


def plot_batch(device: str, *, overwrite: bool) -> None:
    """Generate all batch plots for a device."""
    data = load_batch_data(device)
    if data is None:
        print(f"Warning: No batch data for {device}, skipping...")
        return

    print(f"Plotting batch results - {device}")
    plot_batch_throughput(device, data, overwrite=overwrite)
    plot_batch_latency(device, data, overwrite=overwrite)
    plot_batch_scaling(device, data, overwrite=overwrite)


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
    parser.add_argument(
        "--skip-devices",
        type=str,
        default="",
        help="Comma-separated list of device names to skip.",
    )
    parser.add_argument(
        "--optimizations",
        action="store_true",
        help="Generate per-device optimization speedup bar charts from data/optimizations/.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Generate batch throughput/latency/scaling plots from data/batch/.",
    )
    args = parser.parse_args()

    skip_devices = set()
    if args.skip_devices:
        skip_devices = {d.strip() for d in args.skip_devices.split(",") if d.strip()}

    if args.batch:
        batch_dir = Path(__file__).parent / "data" / "batch"
        if not batch_dir.exists():
            print("Warning: data/batch/ not found, nothing to plot.")
        else:
            devices = [
                f.stem for f in batch_dir.iterdir() if f.is_file() and f.suffix == ".json"
            ]
            if args.device is not None:
                devices = [d for d in devices if d == args.device]
            for device in devices:
                if device in skip_devices:
                    print(f"Skipping device: {device}")
                    continue
                try:
                    plot_batch(device, overwrite=args.overwrite)
                except Exception as e:
                    print(f"Warning: Failed to generate batch plots for {device}: {e}")
    elif args.optimizations:
        opt_dir = Path(__file__).parent / "data" / "optimizations"
        if not opt_dir.exists():
            print("Warning: data/optimizations/ not found, nothing to plot.")
        else:
            devices = [
                f.stem for f in opt_dir.iterdir() if f.is_file() and f.suffix == ".json"
            ]
            if args.device is not None:
                devices = [d for d in devices if d == args.device]
            for device in devices:
                if device in skip_devices:
                    print(f"Skipping device: {device}")
                    continue
                try:
                    plot_optimizations(device, overwrite=args.overwrite)
                except Exception as e:
                    print(f"Warning: Failed to generate optimization plot for {device}: {e}")
    else:
        # parse all the data
        all_data = get_data()
        
        # Load model info for pareto optimal plots
        model_info = get_model_info()

        for name, data in all_data:
            if name in skip_devices:
                print(f"Skipping device: {name}")
                continue
                
            if args.device is None or name == args.device:
                try:
                    if args.latency:
                        plot_device(name, data, overwrite=args.overwrite)
                    if args.pareto:
                        plot_pareto(
                            name, data, model_info, args.framework, overwrite=args.overwrite
                        )
                except Exception as e:
                    print(f"Warning: Failed to generate plots for {name}: {e}, skipping...")
                    continue
