# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: T201
"""Optimization speedup bar charts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from utils.config import DATA_DIR

_PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"

# Seven configs for the bar chart: baseline then CUDA-only configs
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
    Load optimization data and return speedups relative to CPU baseline.

    Returns None if data is missing or invalid.
    """
    path = DATA_DIR / "optimizations" / f"{device}.json"
    if not path.exists():
        return None
    try:
        with path.open("r") as f:
            payload = json.load(f)
    except Exception:
        return None

    if isinstance(payload, list):
        results = payload
    elif isinstance(payload, dict) and "results" in payload:
        results = payload["results"]
    else:
        return None

    # Find the CPU baseline mean latency (support both old "mean_ms" and new "mean" keys)
    baseline_ms: float | None = None
    for _label, prep, cg, pl, um in OPTIMIZATION_BAR_CONFIGS:
        r = _find_result(results, prep, cg, pl, um)
        if r is not None and prep == "cpu":
            baseline_ms = r.get("mean") or r.get("mean_ms")
            break

    if baseline_ms is None or baseline_ms <= 0:
        return None

    speedups: list[float] = []
    for _label, prep, cg, pl, um in OPTIMIZATION_BAR_CONFIGS:
        r = _find_result(results, prep, cg, pl, um)
        if r is not None:
            mean_ms = r.get("mean") or r.get("mean_ms")
            if mean_ms is not None and mean_ms > 0:
                speedups.append(baseline_ms / mean_ms)
            else:
                speedups.append(0.0)
        else:
            speedups.append(0.0)

    return speedups


def plot_optimizations(device: str, *, overwrite: bool) -> None:
    """Plot per-device optimization speedup bar chart."""
    speedups = load_optimization_data(device)
    if speedups is None:
        print(f"Warning: No optimization data for {device}, skipping...")
        return

    plot_dir = _PLOT_DIR / device
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "optimizations.png"
    if plot_path.exists() and not overwrite:
        return

    labels = [c[0] for c in OPTIMIZATION_BAR_CONFIGS]

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
