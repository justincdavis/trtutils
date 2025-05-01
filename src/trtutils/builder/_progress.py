# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from tqdm import tqdm

with contextlib.suppress(ImportError):
    import tensorrt as trt

if TYPE_CHECKING:
    from typing_extensions import Self


class ProgressBar(trt.IProgressMonitor):  # type: ignore[misc]
    """A progress bar for building TensorRT engines."""

    def __init__(self: Self) -> None:
        """Initialize the progress bar."""
        super().__init__()

        self._progress_bars: dict[str, tqdm] = {}
        self._phase_parents: dict[str, str | None] = {}  # Track parent relationships
        self._indentation_levels: dict[
            str,
            int,
        ] = {}  # Track indentation levels directly
        self._last_steps: dict[str, int] = {}  # Track last step for each phase
        self._interrupted: bool = False

    def __del__(self: Self) -> None:
        for progress_bar in self._progress_bars.values():
            progress_bar.close()

    def phase_start(
        self: Self,
        phase_name: str,
        parent_phase: str | None,
        num_steps: int,
    ) -> None:
        """
        Start a new phase.

        Parameters
        ----------
        phase_name : str
            The name of the phase.
        parent_phase : str | None
            The name of the parent phase, or None if the phase is a root phase.
        num_steps : int
            The number of steps in the phase.

        """
        try:
            # Store parent relationship
            self._phase_parents[phase_name] = parent_phase

            # Calculate indentation based on parent
            current_indent = 0
            if parent_phase is not None and parent_phase in self._indentation_levels:
                current_indent = self._indentation_levels[parent_phase] + 1

            self._indentation_levels[phase_name] = current_indent
            self._last_steps[phase_name] = 0  # Initialize last step counter

            # Create progress bar with indentation
            indent = "  " * current_indent
            desc = f"{indent}{phase_name}"
            self._progress_bars[phase_name] = tqdm(
                total=num_steps,
                desc=desc,
                leave=False,
                position=len(self._progress_bars),
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            )
        except KeyboardInterrupt:
            self._interrupted = True

    def step_complete(self: Self, phase_name: str, step: int) -> bool:
        """
        Step in current phase is completed.

        Parameters
        ----------
        phase_name : str
            The name of the phase.
        step : int
            The step number.

        Returns
        -------
        bool
            True if the build should continue, False if it should be interrupted.

        """
        try:
            if phase_name in self._progress_bars:
                last_step = self._last_steps[phase_name]
                step_diff = step - last_step
                if step_diff > 0:
                    self._progress_bars[phase_name].update(step_diff)
                    self._last_steps[phase_name] = step
        except KeyboardInterrupt:
            self._interrupted = True
            return False

        return not self._interrupted

    def phase_finish(self: Self, phase_name: str) -> None:
        """Finish the current phase."""
        try:
            if phase_name in self._progress_bars:
                self._progress_bars[phase_name].close()
                del self._progress_bars[phase_name]
            if phase_name in self._phase_parents:
                del self._phase_parents[phase_name]
            if phase_name in self._indentation_levels:
                del self._indentation_levels[phase_name]
            if phase_name in self._last_steps:
                del self._last_steps[phase_name]
        except KeyboardInterrupt:
            self._interrupted = True
