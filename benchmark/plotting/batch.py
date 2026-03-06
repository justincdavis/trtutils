# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: T201
"""Batch throughput, latency, and scaling plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.config import BATCH_FRAMEWORKS, DATA_DIR

_PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
BATCH_COLORS = {fw: plt.cm.tab10(idx) for idx, fw in enumerate(BATCH_FRAMEWORKS)}


def load_batch_data(
    device: str,
) -> dict[str, dict[str, dict[str, dict[str, float]]]] | None:
    """Load batch benchmark data for a device."""
    path = DATA_DIR / "batch" / f"{device}.json"
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
    plot_dir = _PLOT_DIR / device
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "batch_throughput.png"
    if plot_path.exists() and not overwrite:
        return

    plt.style.use("seaborn-v0_8")
    _, ax = plt.subplots(figsize=(10, 6))
    fontsize = 14

    for fw in BATCH_FRAMEWORKS:
        for _model_name, model_data in data.get(fw, {}).items():
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
        f"{device} - Batch Size vs Throughput",
        fontsize=fontsize + 4,
        fontweight="bold",
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
    plot_dir = _PLOT_DIR / device
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "batch_latency.png"
    if plot_path.exists() and not overwrite:
        return

    plt.style.use("seaborn-v0_8")
    _, ax = plt.subplots(figsize=(10, 6))
    fontsize = 14

    for fw in BATCH_FRAMEWORKS:
        for _model_name, model_data in data.get(fw, {}).items():
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
        f"{device} - Batch Size vs Latency",
        fontsize=fontsize + 4,
        fontweight="bold",
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
    """
    Line plot: batch size vs scaling efficiency.

    efficiency(N) = throughput_at_N / (N * throughput_at_1)
    1.0 = perfect linear scaling.
    """
    plot_dir = _PLOT_DIR / device
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "batch_scaling.png"
    if plot_path.exists() and not overwrite:
        return

    plt.style.use("seaborn-v0_8")
    _, ax = plt.subplots(figsize=(10, 6))
    fontsize = 14

    for fw in BATCH_FRAMEWORKS:
        for _model_name, model_data in data.get(fw, {}).items():
            batch_sizes = sorted(int(k) for k in model_data)
            if not batch_sizes or "1" not in model_data:
                continue

            tp_at_1 = model_data["1"]["throughput"]
            if tp_at_1 <= 0:
                continue

            efficiencies = [model_data[str(bs)]["throughput"] / (bs * tp_at_1) for bs in batch_sizes]

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

    ax.axhline(
        y=1.0,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Perfect scaling",
    )
    ax.set_xlabel("Batch Size", fontsize=fontsize)
    ax.set_ylabel("Scaling Efficiency", fontsize=fontsize)
    ax.set_title(
        f"{device} - Batch Scaling Efficiency",
        fontsize=fontsize + 4,
        fontweight="bold",
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
