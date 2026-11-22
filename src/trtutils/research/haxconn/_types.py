# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Characterization data, calibration, schedules and configuration for HaX-CoNN."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from trtutils.research.shift import reduce_values

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typing_extensions import Self

ACCELERATORS = ("gpu", "dla")
_OBJECTIVES = ("latency", "throughput")
_CONTENTIONS = ("overlap", "static", "none")
_SOLVERS = ("z3", "exhaustive")


@dataclass
class GroupTraits:
    """
    Characterized traits of one layer group of a DNN.

    A layer group is a contiguous range of TensorRT network layers that is always
    mapped to one accelerator. Dictionaries are keyed by accelerator, ``"gpu"`` or
    ``"dla"``; a group without a ``"dla"`` entry in ``time_ms`` cannot run on a DLA.

    Attributes
    ----------
    first : int
        The index of the first network layer of the group.
    last : int
        The index of the last network layer of the group.
    time_ms : dict[str, list[float]]
        Raw standalone compute time samples per accelerator, in milliseconds.
    transition_in_ms : dict[str, float]
        The cost of starting the group on an accelerator after the previous group ran
        elsewhere (input reformat, plus the DLA task launch on a DLA), in milliseconds.
    transition_out_ms : dict[str, float]
        The cost of handing the group's output to the other accelerator, in milliseconds.
    memory : dict[str, list[float]]
        Raw memory throughput samples per accelerator, as EMC utilisation in percent.
        Empty when not measured.

    """

    first: int
    last: int
    time_ms: dict[str, list[float]]
    transition_in_ms: dict[str, float]
    transition_out_ms: dict[str, float]
    memory: dict[str, list[float]] = field(default_factory=dict)

    @property
    def dla(self: Self) -> bool:
        """Whether the group can run on a DLA."""
        return "dla" in self.time_ms

    def time(self: Self, accelerator: str, metric: str = "mean") -> float:
        """
        Get the standalone compute time of the group.

        Parameters
        ----------
        accelerator : str
            ``"gpu"`` or ``"dla"``.
        metric : str, optional
            How to reduce the samples: "mean", "median", "min" or "max". Default is "mean".

        Returns
        -------
        float
            The time in milliseconds.

        """
        return reduce_values(self.time_ms[accelerator], metric)

    def throughput(self: Self, accelerator: str, metric: str = "median") -> float:
        """
        Get the memory throughput of the group.

        Parameters
        ----------
        accelerator : str
            ``"gpu"`` or ``"dla"``.
        metric : str, optional
            How to reduce the samples: "mean", "median", "min" or "max". Default is "median".

        Returns
        -------
        float
            The EMC utilisation in percent, 0.0 when not measured.

        """
        return reduce_values(self.memory.get(accelerator, []), metric)


@dataclass
class ModelTraits:
    """
    Characterized traits of one DNN, from :func:`characterize`.

    Attributes
    ----------
    groups : list[GroupTraits]
        The layer groups, in network order, covering every network layer.
    dla_launch_ms : float
        The DLA task launch cost that was moved from each group's DLA time into its
        DLA transition-in cost, in milliseconds. 0.0 when it could not be measured.
    precision : str
        The precision the groups were measured in, "fp16" or "int8".

    """

    groups: list[GroupTraits]
    dla_launch_ms: float = 0.0
    precision: str = "fp16"

    @property
    def num_layers(self: Self) -> int:
        """The number of network layers."""
        return self.groups[-1].last + 1


@dataclass
class Calibration:
    """
    Contention calibration of one accelerator, from :func:`calibrate`.

    Attributes
    ----------
    accelerator : str
        The victim accelerator, ``"gpu"`` or ``"dla"``.
    demand : list[float]
        The standalone memory throughput of each victim probe (EMC percent), ascending.
    external : list[float]
        The standalone memory throughput of each aggressor probe on the other
        accelerator (EMC percent), ascending.
    relative_speed : list[list[float]]
        Per victim probe and aggressor probe, the victim's speed while both run, in
        percent of its standalone speed.

    """

    accelerator: str
    demand: list[float]
    external: list[float]
    relative_speed: list[list[float]]


class Schedule(NamedTuple):
    """
    An accelerator mapping of every layer group of every DNN.

    Attributes
    ----------
    assignments : dict[str, tuple[str, ...]]
        Per DNN, the accelerator (``"gpu"`` or ``"dla"``) of each layer group.
    latency_ms : dict[str, float]
        Per DNN, the predicted latency when all DNNs start together, in milliseconds.
    objective : float
        The maximum latency in milliseconds (latency objective) or the summed frame
        rate in frames per second (throughput objective).
    optimal : bool
        Whether the solver proved that no schedule is better.

    """

    assignments: dict[str, tuple[str, ...]]
    latency_ms: dict[str, float]
    objective: float
    optimal: bool


@dataclass
class HaxConnConfig:
    """
    Hyperparameters of HaX-CoNN.

    The defaults follow the HaX-CoNN paper. Use :meth:`HaxConnConfig.upstream` for the
    reference implementation's choices.

    Attributes
    ----------
    objective : str
        "latency" minimizes the maximum latency among the DNNs (Eq. 11), "throughput"
        maximizes their summed frame rate (Eq. 10). Default is "latency".
    contention : str
        How shared-memory contention slows layer groups down. "overlap" applies the
        PCCS slowdown only while groups of different DNNs overlap on different
        accelerators (Eq. 7, the paper); "static" always applies the slowdown at
        saturated external demand (reference implementation); "none" ignores
        contention. Default is "overlap".
    max_transitions : int | None
        The maximum number of GPU/DLA transitions per DNN. The paper states no bound
        but every reported schedule has at most one, and the reference implementation
        enforces one. None means unbounded. Default is 1.
    solver : str
        "z3" lets Z3 propose schedules that could beat the best one found and scores
        each with :func:`simulate`; "exhaustive" scores every schedule within
        ``max_transitions``. Default is "z3".
    time_budget : float | None
        Seconds the solver may search before returning the best schedule found.
        None searches until the best schedule is proven optimal. Default is 30.0.
    latency_metric : str
        Reduction of raw time samples: "mean", "median", "min" or "max". Default is "mean".
    memory_metric : str
        Reduction of raw memory throughput samples, same choices. Default is "median".

    """

    objective: str = "latency"
    contention: str = "overlap"
    max_transitions: int | None = 1
    solver: str = "z3"
    time_budget: float | None = 30.0
    latency_metric: str = "mean"
    memory_metric: str = "median"

    def __post_init__(self: Self) -> None:
        choices = {
            "objective": (self.objective, _OBJECTIVES),
            "contention": (self.contention, _CONTENTIONS),
            "solver": (self.solver, _SOLVERS),
        }
        for name, (value, allowed) in choices.items():
            if value not in allowed:
                err_msg = f"Unknown {name} {value!r}, expected one of {allowed}"
                raise ValueError(err_msg)
        reduce_values([0.0], self.latency_metric)
        reduce_values([0.0], self.memory_metric)
        bounds = {
            "max_transitions": self.max_transitions is None or self.max_transitions >= 0,
            "time_budget": self.time_budget is None or self.time_budget > 0.0,
        }
        for name, ok in bounds.items():
            if not ok:
                err_msg = f"{name} is out of range, got {getattr(self, name)}"
                raise ValueError(err_msg)
        if self.solver == "exhaustive" and self.max_transitions is None:
            err_msg = "the exhaustive solver needs max_transitions"
            raise ValueError(err_msg)

    @classmethod
    def upstream(cls: type[Self]) -> Self:
        """
        Get the configuration of the HaX-CoNN reference implementation.

        Returns
        -------
        HaxConnConfig
            The reference implementation's hyperparameters: a static per-group slowdown
            and each group's peak memory throughput.

        """
        return cls(contention="static", memory_metric="max")


def save_traits(traits: ModelTraits, path: Path | str) -> None:
    """
    Save the traits of one DNN to a JSON file.

    Parameters
    ----------
    traits : ModelTraits
        The traits, from :func:`characterize`.
    path : Path | str
        The JSON file to write.

    """
    with Path(path).open("w") as f:
        json.dump(asdict(traits), f, indent=4)


def load_traits(path: Path | str) -> ModelTraits:
    """
    Load traits written by :func:`save_traits`.

    Parameters
    ----------
    path : Path | str
        The JSON file to read.

    Returns
    -------
    ModelTraits
        The traits.

    """
    with Path(path).open() as f:
        data = json.load(f)
    groups = [GroupTraits(**group) for group in data.pop("groups")]
    return ModelTraits(groups=groups, **data)


def save_calibration(calibration: Mapping[str, Calibration], path: Path | str) -> None:
    """
    Save a contention calibration to a JSON file.

    Parameters
    ----------
    calibration : Mapping[str, Calibration]
        The calibration of each accelerator, from :func:`calibrate`.
    path : Path | str
        The JSON file to write.

    """
    with Path(path).open("w") as f:
        json.dump({name: asdict(c) for name, c in calibration.items()}, f, indent=4)


def load_calibration(path: Path | str) -> dict[str, Calibration]:
    """
    Load a calibration written by :func:`save_calibration`.

    Parameters
    ----------
    path : Path | str
        The JSON file to read.

    Returns
    -------
    dict[str, Calibration]
        The calibration of each accelerator.

    """
    with Path(path).open() as f:
        data = json.load(f)
    return {name: Calibration(**entry) for name, entry in data.items()}


def save_schedule(schedule: Schedule, path: Path | str) -> None:
    """
    Save a schedule to a JSON file.

    Parameters
    ----------
    schedule : Schedule
        The schedule.
    path : Path | str
        The JSON file to write.

    """
    with Path(path).open("w") as f:
        json.dump(schedule._asdict(), f, indent=4)


def load_schedule(path: Path | str) -> Schedule:
    """
    Load a schedule written by :func:`save_schedule`.

    Parameters
    ----------
    path : Path | str
        The JSON file to read.

    Returns
    -------
    Schedule
        The schedule.

    """
    with Path(path).open() as f:
        data = json.load(f)
    assignments = {name: tuple(accs) for name, accs in data["assignments"].items()}
    return Schedule(assignments, data["latency_ms"], data["objective"], data["optimal"])
