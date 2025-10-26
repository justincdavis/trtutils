# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
PID controller and plan controller for runtime adaptation.

Classes
-------
PIDController
    Standard PID controller.
PlanController
    Selects partition plans using PID-adjusted budget.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from trtutils._log import LOG

if TYPE_CHECKING:
    from typing_extensions import Self

    from ._plan import PartitionPlan


class PIDController:
    """Standard PID controller for feedback control."""

    def __init__(
        self: Self,
        kp: float = 0.6,
        ki: float = 0.3,
        kd: float = 0.1,
    ) -> None:
        """
        Initialize PID controller.

        Parameters
        ----------
        kp : float, optional
            Proportional gain, by default 0.6.
        ki : float, optional
            Integral gain, by default 0.3.
        kd : float, optional
            Derivative gain, by default 0.1.

        """
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.prev_error = 0.0
        self.integral = 0.0

    def update(self: Self, error: float) -> float:
        """
        Compute control signal from error.

        Parameters
        ----------
        error : float
            Current error (target - actual).

        Returns
        -------
        float
            Control signal.

        """
        # Update integral
        self.integral += error

        # Compute derivative
        derivative = error - self.prev_error
        self.prev_error = error

        # Compute control signal
        control = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        return control

    def reset(self: Self) -> None:
        """Reset controller state."""
        self.prev_error = 0.0
        self.integral = 0.0


class PlanController:
    """Controls partition plan selection using PID feedback."""

    def __init__(
        self: Self,
        plan_pool: list[PartitionPlan],
        pid: PIDController,
        initial_budget: float,
    ) -> None:
        """
        Initialize plan controller.

        Parameters
        ----------
        plan_pool : list[PartitionPlan]
            Available partition plans.
        pid : PIDController
            PID controller for feedback.
        initial_budget : float
            Initial latency budget in seconds.

        """
        self.plan_pool = plan_pool
        self.pid = pid
        self.budget = initial_budget

        # Select initial plan closest to budget
        self.current_plan = self._select_best_plan(initial_budget)

    def adjust(
        self: Self,
        actual_latency: float,
        *,
        verbose: bool = False,
    ) -> PartitionPlan:
        """
        Adjust plan selection based on actual latency.

        Parameters
        ----------
        actual_latency : float
            Measured latency from last frame in seconds.
        verbose : bool, optional
            Whether to log information, by default False.

        Returns
        -------
        PartitionPlan
            Selected plan for next frame.

        """
        # Compute error
        error = self.budget - actual_latency

        # Update PID
        adjustment = self.pid.update(error)

        # Adjust target budget
        target_budget = self.budget + adjustment

        # Select best plan for adjusted budget
        new_plan = self._select_best_plan(target_budget)

        if verbose and new_plan.plan_id != self.current_plan.plan_id:
            LOG.debug(
                f"Plan switch: {self.current_plan.plan_id} -> {new_plan.plan_id} "
                f"(budget={target_budget*1000:.2f}ms, "
                f"actual={actual_latency*1000:.2f}ms)",
            )

        self.current_plan = new_plan
        return self.current_plan

    def _select_best_plan(self: Self, budget: float) -> PartitionPlan:
        """
        Select best plan for given budget.

        Parameters
        ----------
        budget : float
            Target latency budget in seconds.

        Returns
        -------
        PartitionPlan
            Best plan within budget.

        """
        # Filter plans within budget
        viable = [p for p in self.plan_pool if p.est_lat <= budget]

        if len(viable) == 0:
            # No plan within budget, use fastest
            return min(self.plan_pool, key=lambda p: p.est_lat)

        # Select highest accuracy among viable plans
        return max(viable, key=lambda p: p.est_ap)

    def get_current_plan(self: Self) -> PartitionPlan:
        """
        Get current partition plan.

        Returns
        -------
        PartitionPlan
            Current plan.

        """
        return self.current_plan

    def reset(self: Self) -> None:
        """Reset controller state."""
        self.pid.reset()
        self.current_plan = self._select_best_plan(self.budget)

