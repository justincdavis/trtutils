# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: T201
"""
Latency and throughput across a series of patch stages.

Unlike the other plotting modules, which compare frameworks on a fixed
checkout, this one compares a single framework across successive commits.
Each stage is one commit of a patch series, benchmarked with the harness
held constant so only the library under test varies. Stage files are
written by ``run.py optimize --stage`` / ``run.py batch --stage`` to
``data/perf-series/<device>/stage-<NAME>.json``; each command owns one
section (``optimize`` / ``batch``) of that file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from collections.abc import Sequence

_PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"

PREPROCESSORS = ("cpu", "cuda", "trt")
COMPOSITIONS = ("homogeneous8", "heterogeneous8", "resolution-switch")
# the raw tensorrt path runs the same engine without the library in the
# loop, so it is a control for machine noise and should stay flat
BATCH_FRAMEWORK = "trtutils(graph)"
BATCH_CONTROL = "tensorrt(graph)"
PREP_COLORS = {"cpu": plt.cm.tab10(0), "cuda": plt.cm.tab10(1), "trt": plt.cm.tab10(2)}


def load_stages(results_dir: Path) -> list[dict]:
    """Load every stage result file, ordered by stage name."""
    stages = []
    for path in sorted(results_dir.glob("stage-*.json")):
        with path.open("r") as f:
            stages.append(json.load(f))
    return stages


def _optimize_series(
    stages: list[dict],
    preprocessor: str,
    inputs: str,
    metric: str,
    *,
    cuda_graph: bool = True,
) -> tuple[list[int], list[float]]:
    """One optimize-grid metric across stages (pagelocked, non-unified rows)."""
    xs: list[int] = []
    ys: list[float] = []
    for idx, stage in enumerate(stages):
        for row in stage.get("optimize", {}).get("results", []):
            if (
                row["preprocessor"] == preprocessor
                and row["inputs"] == inputs
                and row["cuda_graph"] == cuda_graph
                and row["pagelocked_mem"]
                and not row["unified_mem"]
            ):
                xs.append(idx)
                ys.append(row[metric])
                break
    return xs, ys


def _batch_series(
    stages: list[dict],
    framework: str,
    model: str,
    key: str,
) -> tuple[list[int], list[float]]:
    """Batch-sweep throughput for one framework/batch across stages."""
    xs: list[int] = []
    ys: list[float] = []
    for idx, stage in enumerate(stages):
        entry = stage.get("batch", {}).get("data", {}).get(framework, {}).get(model, {}).get(key)
        if entry:
            xs.append(idx)
            ys.append(entry["throughput"])
    return xs, ys


def _annotate_series(
    ax: plt.Axes,
    n_stages: int,
    series_boundaries: Sequence[tuple[float, str]],
) -> None:
    for pos, label in series_boundaries:
        if pos > n_stages - 1:
            continue
        ax.axvline(pos, color="0.4", linestyle=":", linewidth=1.2, zorder=0)
        ax.annotate(
            label,
            xy=(pos, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(4, -4),
            textcoords="offset points",
            fontsize=7,
            color="0.35",
            va="top",
            ha="left",
        )


def plot_patch_series(
    results_dir: Path,
    output: Path | None = None,
    *,
    batch_model: str = "yolov10n",
    series_boundaries: Sequence[tuple[float, str]] = (),
) -> Path:
    """
    Plot latency and throughput across every stage of a patch series.

    Parameters
    ----------
    results_dir : Path
        Directory of stage-NN.json result files, e.g.
        data/perf-series/<device>.
    output : Path, optional
        Where to write the PNG. Defaults to plots/patch_series.png.
    batch_model : str
        The model to use for the batch-throughput panel.
    series_boundaries : Sequence[tuple[float, str]]
        (stage index - 0.5, label) pairs marking boundaries between
        independently developed patch series. Empty by default.

    Returns
    -------
    Path
        The path the plot was written to.

    Raises
    ------
    FileNotFoundError
        If no stage result files are found.

    """
    stages = load_stages(results_dir)
    if not stages:
        err_msg = f"No stage-*.json files found in {results_dir}"
        raise FileNotFoundError(err_msg)

    labels = [str(s["stage"]) for s in stages]
    fig, axes = plt.subplots(3, 1, figsize=(11, 13), sharex=True)
    ax_single, ax_comp, ax_batch = axes

    # --- panel 1: single image end2end latency, per preprocessor -----------
    for prep in PREPROCESSORS:
        for cuda_graph, alpha in ((True, 1.0), (False, 0.45)):
            xs, ys = _optimize_series(stages, prep, "single", "mean", cuda_graph=cuda_graph)
            if xs:
                ax_single.plot(
                    xs,
                    ys,
                    marker="o",
                    markersize=3.5,
                    linewidth=1.6,
                    color=PREP_COLORS[prep],
                    alpha=alpha,
                    label=f"{prep}{' (graph)' if cuda_graph else ''}",
                )
    ax_single.set_ylabel("latency (ms)")
    ax_single.set_title("single image end2end — lower is better", fontsize=10, loc="left")

    # --- panel 2: composition throughput, graph on ---------------------------
    styles = {"homogeneous8": "-", "heterogeneous8": "--", "resolution-switch": ":"}
    for prep in PREPROCESSORS:
        for comp in COMPOSITIONS:
            xs, ys = _optimize_series(stages, prep, comp, "throughput")
            if xs:
                ax_comp.plot(
                    xs,
                    ys,
                    marker="o",
                    markersize=3.5,
                    linewidth=1.6,
                    linestyle=styles[comp],
                    color=PREP_COLORS[prep],
                    label=f"{prep} {comp}",
                )
    ax_comp.set_ylabel("throughput (img/s)")
    ax_comp.set_title("input compositions (graph) — higher is better", fontsize=10, loc="left")

    # --- panel 3: batch sweep ------------------------------------------------
    batch_keys = sorted(
        {
            k
            for s in stages
            for k in s.get("batch", {}).get("data", {}).get(BATCH_FRAMEWORK, {}).get(batch_model, {})
        },
        key=int,
    )
    cmap = plt.cm.viridis
    for idx, key in enumerate(batch_keys):
        shade = cmap(idx / max(len(batch_keys) - 1, 1))
        xs, ys = _batch_series(stages, BATCH_FRAMEWORK, batch_model, key)
        if xs:
            ax_batch.plot(
                xs, ys, marker="o", markersize=3.5, linewidth=1.6, color=shade, label=f"batch {key}"
            )
        xs, ys = _batch_series(stages, BATCH_CONTROL, batch_model, key)
        if xs:
            ax_batch.plot(xs, ys, linestyle="--", linewidth=1.0, color=shade, alpha=0.35)
    ax_batch.set_ylabel("throughput (img/s)")
    ax_batch.set_title(
        f"batch sweep — {batch_model} {BATCH_FRAMEWORK} (dashed = raw TensorRT control) — higher is better",
        fontsize=10,
        loc="left",
    )
    ax_batch.set_xticks(range(len(stages)))
    ax_batch.set_xticklabels(labels)
    ax_batch.set_xlabel("stage")

    for ax in axes:
        ax.grid(visible=True, alpha=0.25)
        ax.legend(fontsize=7, ncol=3, loc="best", framealpha=0.85)
        _annotate_series(ax, len(stages), series_boundaries)

    fig.suptitle("trtutils perf series — per-stage latency and throughput", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output = output or (_PLOT_DIR / "patch_series.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"wrote {output}")
    return output
