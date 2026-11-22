#!/usr/bin/env python3
# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Total-coverage ratchet for the CPU test suite run by GitHub Actions.

Reads a coverage.py JSON report and compares its total coverage
(``totals.percent_covered``, which counts lines and branches together when
branch coverage is on) against the committed baseline in
``.coverage-baseline.cpu.json``. Coverage may never go down: the check fails
when the total is more than the tolerance (default 0.01 points, for float
rounding) below the baseline.

On pull requests CI also passes the base branch's copy of the baseline with
``--base-baseline``. The floor is then the higher of the two baselines, so a
PR cannot get past the check by lowering the committed number, and the check
fails if it tries to.

When coverage goes up, the check passes and says so; run with ``--update``
to raise the committed baseline to the new total so it ratchets up. The
baseline never moves down.

This script does NOT run tests. ``make test-cpu-cov`` runs the CPU suite
with coverage (writing ``.coverage-cpu.json``) and then this check.

Usage::

    python ci/check_coverage_total.py                      # Check mode
    python ci/check_coverage_total.py --update             # Ratchet baseline up
    python ci/check_coverage_total.py --report FILE        # Report to read
    python ci/check_coverage_total.py --baseline FILE      # Baseline to check/update
    python ci/check_coverage_total.py --base-baseline FILE # Base branch baseline
    python ci/check_coverage_total.py --tolerance 0.05     # Override tolerance

Exit codes:
    0 -- Coverage at or above the baseline
    1 -- Coverage below the baseline, or the PR lowers the baseline
    2 -- Missing or malformed report or baseline file
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT = ROOT / ".coverage-cpu.json"
DEFAULT_BASELINE = ROOT / ".coverage-baseline.cpu.json"
DEFAULT_TOLERANCE = 0.01
KEY = "percent_covered"
PRECISION = 2


class _InputError(Exception):
    """A report or baseline file is missing or malformed."""


def _read_json(path: Path, what: str) -> dict:
    if not path.exists():
        msg = f"{what} not found: {path}"
        raise _InputError(msg)
    try:
        with path.open() as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        msg = f"{what} is not valid JSON: {path} ({e})"
        raise _InputError(msg) from e
    if not isinstance(data, dict):
        msg = f"{what} is not a JSON object: {path}"
        raise _InputError(msg)
    return data


def _report_total(path: Path) -> float:
    totals = _read_json(path, "Coverage report").get("totals", {})
    if KEY not in totals:
        msg = f"Coverage report has no totals.{KEY}: {path}"
        raise _InputError(msg)
    return round(float(totals[KEY]), PRECISION)


def _baseline_total(path: Path) -> float:
    data = _read_json(path, "Coverage baseline")
    if KEY not in data:
        msg = f"Coverage baseline has no {KEY}: {path}"
        raise _InputError(msg)
    return round(float(data[KEY]), PRECISION)


def _below(value: float, floor: float, tol: float) -> bool:
    """Return whether ``value`` is more than ``tol`` points below ``floor``."""
    # round the gap so float error (29.96 - 0.01 != 29.95) cannot flip the result
    return round(floor - value, PRECISION + 2) > tol


def _annotate(level: str, message: str) -> None:
    """Print ``message``, as a workflow annotation when running in GitHub Actions."""
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::{level} title=Coverage ratchet::{message}")
    else:
        print(f"{level.upper()}: {message}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail if total coverage drops below the committed baseline.",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--base-baseline",
        type=Path,
        default=None,
        help="the base branch's baseline file (pull requests)",
    )
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument(
        "--update",
        action="store_true",
        help="raise the baseline to the current total (never lowers it)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Check total coverage against the baseline, or ratchet the baseline up."""
    args = _parse_args(argv)
    tol = args.tolerance

    try:
        current = _report_total(args.report)
        baseline = _baseline_total(args.baseline)
        base = _baseline_total(args.base_baseline) if args.base_baseline else None
    except _InputError as e:
        print(f"ERROR: {e}")
        return 2

    print(f"Current total coverage:  {current:.{PRECISION}f}%")
    print(f"Committed baseline:      {baseline:.{PRECISION}f}% ({args.baseline.name})")
    floor = baseline
    failed = False
    if base is not None:
        print(f"Base branch baseline:    {base:.{PRECISION}f}%")
        if _below(baseline, base, tol):
            _annotate(
                "error",
                f"the committed baseline {baseline:.{PRECISION}f}% is lower than the base "
                f"branch's {base:.{PRECISION}f}%; the baseline may only go up",
            )
            failed = True
        floor = max(baseline, base)

    if _below(current, floor, tol):
        _annotate(
            "error",
            f"total coverage fell from {floor:.{PRECISION}f}% to {current:.{PRECISION}f}% "
            f"(tolerance {tol}); add tests for the new or changed code",
        )
        failed = True
    elif _below(floor, current, tol):
        _annotate(
            "notice",
            f"total coverage rose from {floor:.{PRECISION}f}% to {current:.{PRECISION}f}%; "
            "run `python ci/check_coverage_total.py --update` on the CI report and commit "
            f"{DEFAULT_BASELINE.name} to raise the baseline",
        )
    else:
        print(f"OK: total coverage matches the baseline ({floor:.{PRECISION}f}%)")

    if args.update and current > baseline:
        with args.baseline.open("w") as f:
            json.dump({KEY: current}, f, indent=4)
            f.write("\n")
        print(f"Baseline raised {baseline:.{PRECISION}f}% -> {current:.{PRECISION}f}%")

    if failed:
        print("\nCoverage ratchet FAILED.")
        return 1
    print("\nCoverage ratchet passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
