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
``data/perf-series/<device>/stage-<NAME>.json``.
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

# the trtutils paths are what the patches change; the raw tensorrt paths run
# the same engine without the library in the loop, so they act as a control
# for machine noise and should stay flat across stages
TRTUTILS_FRAMEWORKS = ("trtutils", "trtutils(graph)")
CONTROL_FRAMEWORKS = ("tensorrt", "tensorrt(graph)")

# boundaries between independently developed patch series, as (stage index -
# 0.5, label) pairs. Empty by default: dnnkit's boundaries are specific to
# its own 14-stage GB10 run and do not apply here. Pass series_boundaries
# (or --boundary on the CLI) to annotate a run's own series.
SERIES_BOUNDARIES: tuple[tuple[float, str], ...] = ()

MODEL_COLORS = {
    "yolov10n": plt.cm.tab10(0),
    "yolov8n": plt.cm.tab10(1),
    "rtdetrv2_r18": plt.cm.tab10(2),
}


def load_stages(results_dir: Path) -> list[dict]:
    """Load every stage result file, ordered by stage index."""
    stages = []
    for path in sorted(results_dir.glob("stage-*.json")):
        with path.open("r") as f:
            stages.append(json.load(f))
    return stages


def _series(
    stages: list[dict],
    section: str,
    framework: str,
    model: str,
    key: str,
    metric: str,
) -> tuple[list[int], list[float]]:
    """Pull one metric across stages, skipping stages with no measurement."""
    xs: list[int] = []
    ys: list[float] = []
    for idx, stage in enumerate(stages):
        data = stage.get(section)
        if not data:
            continue
        entry = data.get(framework, {}).get(model, {}).get(key)
        if not entry:
            continue
        xs.append(idx)
        ys.append(entry[metric])
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
    series_boundaries: Sequence[tuple[float, str]] = SERIES_BOUNDARIES,
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

    labels = [s["stage"].replace("stage-", "") for s in stages]
    models = sorted(
        {
            m
            for s in stages
            if s.get("models")
            for fw in s["models"]
            for m in s["models"][fw]
        }
    )

    fig, axes = plt.subplots(3, 1, figsize=(11, 13), sharex=True)
    ax_lat, ax_tp, ax_batch = axes

    # --- panels 1 and 2: batch-1 latency and throughput -----------------
    for ax, metric, ylabel, better in (
        (ax_lat, "mean", "latency (ms)", "lower is better"),
        (ax_tp, "throughput", "throughput (img/s)", "higher is better"),
    ):
        for model in models:
            color = MODEL_COLORS.get(model, "0.5")
            for framework in TRTUTILS_FRAMEWORKS:
                xs, ys = _series(stages, "models", framework, model, "640", metric)
                if not xs:
                    continue
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    markersize=3.5,
                    linewidth=1.6,
                    color=color,
                    alpha=1.0 if framework == "trtutils" else 0.55,
                    label=f"{model} {framework}",
                )
            for framework in CONTROL_FRAMEWORKS:
                xs, ys = _series(stages, "models", framework, model, "640", metric)
                if not xs:
                    continue
                ax.plot(
                    xs,
                    ys,
                    linestyle="--",
                    linewidth=1.0,
                    color=color,
                    alpha=0.35,
                    label=f"{model} {framework} (control)",
                )
        ax.set_ylabel(ylabel)
        ax.set_title(f"batch 1 @ 640 — {better}", fontsize=10, loc="left")
        ax.grid(visible=True, alpha=0.25)
        _annotate_series(ax, len(stages), series_boundaries)

    # --- panel 3: batch throughput -------------------------------------
    batch_keys = sorted(
        {
            k
            for s in stages
            if s.get("batch")
            for fw in s["batch"]
            for k in s["batch"][fw].get(batch_model, {})
        },
        key=int,
    )
    cmap = plt.cm.viridis
    for idx, key in enumerate(batch_keys):
        shade = cmap(idx / max(len(batch_keys) - 1, 1))
        xs, ys = _series(stages, "batch", "trtutils(graph)", batch_model, key, "throughput")
        if not xs:
            continue
        ax_batch.plot(
            xs,
            ys,
            marker="o",
            markersize=3.5,
            linewidth=1.6,
            color=shade,
            label=f"batch {key}",
        )
    ax_batch.set_ylabel("throughput (img/s)")
    ax_batch.set_title(
        f"batch sweep — {batch_model} trtutils(graph) — higher is better",
        fontsize=10,
        loc="left",
    )
    ax_batch.grid(visible=True, alpha=0.25)
    _annotate_series(ax_batch, len(stages), series_boundaries)

    # mark where the engine flavor changes
    for idx, stage in enumerate(stages):
        if idx and stage.get("engines_rebuilt"):
            for ax in axes:
                ax.axvline(idx, color="crimson", linestyle="-", linewidth=0.8, alpha=0.3)

    ax_batch.set_xticks(range(len(stages)))
    ax_batch.set_xticklabels(labels)
    ax_batch.set_xlabel("patch stage")

    for ax in axes:
        ax.legend(fontsize=7, ncol=2, loc="best", framealpha=0.85)

    fig.suptitle(
        "trtutils patch series — latency and throughput per applied patch\n"
        "dashed = raw TensorRT control (should stay flat); red line = engines rebuilt",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output = output or (_PLOT_DIR / "patch_series.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"wrote {output}")
    return output


if __name__ == "__main__":
    import sys

    results = Path(sys.argv[1])
    dest = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    plot_patch_series(results, dest)
