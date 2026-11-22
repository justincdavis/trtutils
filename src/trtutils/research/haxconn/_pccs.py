# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""PCCS, the processor-centric contention slowdown model HaX-CoNN predicts contention with."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self

    from ._types import Calibration

_MINOR_DROP = 10.0
_MIN_SPEED = 5.0
_FIT_STEPS = 100


@dataclass(frozen=True)
class PCCSModel:
    """
    The PCCS slowdown model of one accelerator (Xu et al., MICRO 2021, Eqs. 1-5).

    The speed of a kernel running on the accelerator depends on its own memory
    throughput demand ``x`` and the total demand ``y`` of the kernels running on the
    other accelerators, both standalone values in the same unit (EMC percent here).
    Kernels up to ``normal_bw`` are in the minor contention region, up to
    ``intensive_bw`` in the normal region, and above that in the intensive region.

    Attributes
    ----------
    normal_bw : float
        Demand separating the minor and normal regions; 0.0 when there is no minor
        region (the DLA).
    intensive_bw : float
        Demand separating the normal and intensive regions.
    mrmc : float
        Maximum slowdown in the minor region, in percent of standalone speed.
    cbp : float
        External demand at which the speed stops dropping (contention balance point).
    tbwdc : float
        Total demand (own plus external) at which the speed starts dropping.
    rate : float
        Speed lost per unit of demand in the normal region, in percent.
    peak : float
        Peak demand of the system (PBW). 100.0 for EMC percent. Default is 100.0.

    """

    normal_bw: float
    intensive_bw: float
    mrmc: float
    cbp: float
    tbwdc: float
    rate: float
    peak: float = 100.0

    @staticmethod
    def from_calibration(calibration: Calibration) -> PCCSModel:
        """
        Fit the model to a calibration from :func:`calibrate`.

        Parameters
        ----------
        calibration : Calibration
            The calibration of one accelerator.

        Returns
        -------
        PCCSModel
            The fitted model.

        """
        return fit_pccs(calibration.demand, calibration.external, calibration.relative_speed)

    def relative_speed(self: Self, x: float, y: float) -> float:
        """
        Predict the relative speed of a kernel under contention.

        Parameters
        ----------
        x : float
            The kernel's standalone memory throughput demand.
        y : float
            The total standalone demand of the kernels on the other accelerators.

        Returns
        -------
        float
            The speed in percent of the standalone speed, in [5, 100]; 100 without
            external demand.

        """
        if y <= 0.0:
            return 100.0
        minor = 100.0 - self.mrmc * x / self.peak
        if x <= self.normal_bw:
            return _clamp(minor)
        rate = self.rate
        if x >= self.intensive_bw and self.cbp > 0.0:
            # eq. 4: the intensive rate grows with the demand
            rate = self.rate * max(x + self.cbp - self.tbwdc, 0.0) / self.cbp
        if y >= self.cbp:
            return _clamp(100.0 - max(x + self.cbp - self.tbwdc, 0.0) * rate)
        if x + y >= self.tbwdc:
            return _clamp(100.0 - (x + y - self.tbwdc) * rate)
        return _clamp(minor)

    def slowdown(self: Self, x: float, y: float) -> float:
        """
        Predict the slowdown factor of a kernel under contention.

        Parameters
        ----------
        x : float
            The kernel's standalone memory throughput demand.
        y : float
            The total standalone demand of the kernels on the other accelerators.

        Returns
        -------
        float
            The time multiplier, at least 1.0.

        """
        return 100.0 / self.relative_speed(x, y)


def _clamp(speed: float) -> float:
    return min(100.0, max(_MIN_SPEED, speed))


def fit_pccs(
    demand: Sequence[float],
    external: Sequence[float],
    relative_speed: Sequence[Sequence[float]],
    peak: float = 100.0,
    tolerance: float = 2.0,
) -> PCCSModel:
    """
    Fit a :class:`PCCSModel` to a calibration matrix.

    The minor and intensive regions follow the PCCS paper's steps 1 and 2. TBWDC, CBP
    and the rate (its steps 3-5) are fit by least squares to Eqs. 3-5 on the normal and
    intensive regions' measurements instead of its turn-point heuristics, which need a
    fine and noise-free grid.

    Parameters
    ----------
    demand : Sequence[float]
        The standalone demand of each victim kernel, ascending.
    external : Sequence[float]
        Each external demand the victims ran against, ascending.
    relative_speed : Sequence[Sequence[float]]
        Per victim and external demand, the victim's speed in percent of standalone.
    peak : float, optional
        The peak demand of the system. Default is 100.0 (EMC percent).
    tolerance : float, optional
        The smallest speed drop, in percent, that counts as a slowdown when the
        accelerator has no minor region. Default is 2.0.

    Returns
    -------
    PCCSModel
        The fitted model. Regions the data does not reach are left empty: with no
        slowdown at all the model never slows anything down.

    Raises
    ------
    ValueError
        If the matrix shape does not match ``demand`` and ``external``, or either
        is not ascending.

    """
    speed = np.asarray(relative_speed, dtype=np.float64)
    x = np.asarray(demand, dtype=np.float64)
    y = np.asarray(external, dtype=np.float64)
    if speed.shape != (len(x), len(y)) or len(x) == 0 or len(y) == 0:
        err_msg = f"relative_speed must be {len(x)}x{len(y)}, got shape {speed.shape}"
        raise ValueError(err_msg)
    if np.any(np.diff(x) < 0) or np.any(np.diff(y) < 0):
        err_msg = "demand and external must be ascending"
        raise ValueError(err_msg)
    drop = 100.0 - speed
    rows = len(x)

    # step 1: the minor region, where even the largest external demand barely slows down
    normal_bw, mrmc, first_normal = 0.0, 0.0, 0
    if drop[0, -1] <= _MINOR_DROP:
        minor_threshold = max(2.0 * drop[0, -1], tolerance)
        first_normal = next((i for i in range(rows) if drop[i, -1] > minor_threshold), rows)
        normal_bw = float(x[first_normal - 1])
        # eq. 2: the drop grows with the demand up to mrmc at the peak
        mrmc = max(
            (float(drop[r, -1]) * peak / float(x[r]) for r in range(first_normal) if x[r] > 0),
            default=0.0,
        )
    if first_normal == rows:
        return PCCSModel(normal_bw, normal_bw, mrmc, peak, 2.0 * peak, 0.0, peak)
    # a slowdown is a drop beyond anything the minor region shows (eq. 2 at normal_bw)
    threshold = max(2.0 * mrmc * normal_bw / peak, tolerance)

    # step 2: the intensive region, which slows down at the smallest external demand
    first_intensive = next((i for i in range(first_normal, rows) if drop[i, 0] >= threshold), rows)
    intensive_bw = float(x[first_intensive]) if first_intensive < rows else peak

    # steps 3-5: TBWDC, CBP and the rate, by least squares on the normal and intensive
    # regions (eqs. 3-5) over a grid of TBWDC and CBP; the rate is linear and solved in
    # closed form, scaled by eq. 4 on intensive rows
    rx = x[first_normal:][:, None]
    ry = y[None, :]
    observed = drop[first_normal:]
    intensive = rx >= intensive_bw
    best: tuple[float, float, float, float] | None = None
    for cbp in np.linspace(0.0, peak, _FIT_STEPS + 1):
        for tbwdc in np.linspace(0.0, 2.0 * peak, 2 * _FIT_STEPS + 1):
            flat = ry >= cbp
            dropping = ~flat & (rx + ry >= tbwdc)
            slope = np.where(flat, np.maximum(rx + cbp - tbwdc, 0.0), 0.0)
            slope = np.where(dropping, rx + ry - tbwdc, slope)
            if cbp > 0.0:
                slope = slope * np.where(intensive, np.maximum(rx + cbp - tbwdc, 0.0) / cbp, 1.0)
            minor = np.where(flat | dropping, 0.0, mrmc * rx / peak) * np.ones_like(observed)
            norm = float(np.sum(slope * slope))
            rate = max(float(np.sum(slope * (observed - minor))) / norm, 0.0) if norm > 0 else 0.0
            error = float(np.sum((observed - minor - rate * slope) ** 2))
            if best is None or error < best[0]:
                best = (error, float(cbp), float(tbwdc), rate)
    _, cbp, tbwdc, rate = best if best is not None else (0.0, peak, 2.0 * peak, 0.0)
    return PCCSModel(normal_bw, intensive_bw, mrmc, cbp, tbwdc, rate, peak)
