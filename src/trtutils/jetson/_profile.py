# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import contextlib
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import TYPE_CHECKING

from jetsontools import TegraData, TegraStats, filter_data, get_powerdraw

with contextlib.suppress(ImportError):
    import tensorrt as trt

from trtutils._benchmark import Metric
from trtutils._engine import TRTEngine
from trtutils._log import LOG
from trtutils.inspect._profiler import LayerTiming, ProfilerResult

if TYPE_CHECKING:
    from jetsontools._parsing import Metric as JMetric
    from typing_extensions import Self


@dataclass
class JetsonLayerTiming(LayerTiming):
    """
    A dataclass to store per-layer profiling statistics for Jetson devices.

    Extends LayerTiming with power and energy metrics.

    Attributes
    ----------
    name : str
        The name of the layer.
    mean : float
        The mean execution time in milliseconds.
    median : float
        The median execution time in milliseconds.
    min : float
        The minimum execution time in milliseconds.
    max : float
        The maximum execution time in milliseconds.
    raw : list[float]
        The raw execution times in milliseconds across all iterations.
    power : float
        The mean power draw in milliwatts during layer execution.
    energy : float
        The mean energy consumption in millijoules per layer execution.

    """

    power: float
    energy: float

    def __str__(self: Self) -> str:
        return (
            f"{self.name}: mean={self.mean:.3f}ms, median={self.median:.3f}ms, "
            f"min={self.min:.3f}ms, max={self.max:.3f}ms, "
            f"power={self.power:.1f}mW, energy={self.energy:.3f}mJ"
        )

    def __repr__(self: Self) -> str:
        return (
            f"JetsonLayerTiming(name={self.name!r}, mean={self.mean}, median={self.median}, "
            f"min={self.min}, max={self.max}, power={self.power}, energy={self.energy})"
        )


@dataclass
class JetsonProfilerResult(ProfilerResult):
    """
    A dataclass to store the complete profiling results for Jetson devices.

    This extends the standard profiling results with energy and power metrics.

    Attributes
    ----------
    layers : list[JetsonLayerTiming]
        The per-layer timing, power, and energy statistics.
    total_time : LayerTiming
        The total execution time statistics across all layers.
    iterations : int
        The number of profiling iterations performed.
    power_draw : Metric
        The power draw statistics in milliwatts.
    energy : Metric
        The energy consumption statistics in milliwatt-seconds.

    """

    power_draw: Metric
    energy: Metric

    def __str__(self: Self) -> str:
        return (
            f"JetsonProfilerResult(layers={len(self.layers)}, "
            f"total_time={self.total_time.mean:.3f}ms, "
            f"iterations={self.iterations}, "
            f"power_draw={self.power_draw.mean:.1f}mW, "
            f"energy={self.energy.mean:.3f}mJ)"
        )

    def __repr__(self: Self) -> str:
        return (
            f"JetsonProfilerResult(layers={self.layers!r}, "
            f"total_time={self.total_time!r}, "
            f"iterations={self.iterations}, "
            f"power_draw={self.power_draw!r}, "
            f"energy={self.energy!r})"
        )


class JetsonLayerProfiler(trt.IProfiler):  # type: ignore[misc]
    """
    A profiler for Jetson devices that tracks per-layer timing and power/energy metrics.

    This class collects per-layer execution times and timestamps across multiple
    inference iterations, then correlates these with tegrastats data to compute
    per-layer power and energy consumption.
    """

    def __init__(self: Self) -> None:
        """Initialize the JetsonLayerProfiler."""
        super().__init__()
        # Store timings for each layer across iterations
        # Key: layer_name, Value: list of timings in milliseconds
        self._timings: dict[str, list[float]] = defaultdict(list)
        # Store timestamps for each layer across iterations
        # Key: layer_name, Value: list of (start_time, end_time) tuples
        self._layer_timestamps: dict[str, list[tuple[float, float]]] = defaultdict(list)
        # Store current iteration timings and timestamps
        self._current_iteration_timings: dict[str, float] = {}
        self._current_iteration_start_time: float = 0.0
        self._current_layer_start_time: float = 0.0

    def start_iteration(self: Self) -> None:
        """
        Mark the start of a new iteration.

        This should be called before each inference run to set the baseline timestamp.
        """
        self._current_iteration_start_time = time.time()
        self._current_layer_start_time = self._current_iteration_start_time

    def report_layer_time(self: Self, layer_name: str, ms: float) -> None:
        """
        Record the execution time for a layer.

        This method is called by TensorRT once per layer after inference.

        Parameters
        ----------
        layer_name : str
            The name of the layer.
        ms : float
            The execution time in milliseconds.

        """
        # Record the timing
        self._current_iteration_timings[layer_name] = ms
        
        # Calculate layer end time
        # ms is in milliseconds, so convert to seconds
        layer_end_time = self._current_layer_start_time + (ms / 1000.0)
        
        # Store timestamp range for this layer
        if layer_name not in self._current_iteration_timings:
            self._current_iteration_timings[layer_name] = ms
        
        # Update for next layer
        self._current_layer_start_time = layer_end_time

    def finalize_iteration(self: Self) -> None:
        """
        Finalize the current iteration by storing all layer timings and timestamps.

        This should be called after inference to commit the current iteration's data.
        """
        for layer_name, time_ms in self._current_iteration_timings.items():
            self._timings[layer_name].append(time_ms)
        
        # Reconstruct timestamps for all layers based on cumulative timing
        current_time = self._current_iteration_start_time
        for layer_name, time_ms in self._current_iteration_timings.items():
            layer_start = current_time
            layer_end = current_time + (time_ms / 1000.0)  # Convert ms to seconds
            self._layer_timestamps[layer_name].append((layer_start, layer_end))
            current_time = layer_end
        
        self._current_iteration_timings.clear()

    def get_statistics(self: Self) -> list[LayerTiming]:
        """
        Compute basic timing statistics for each layer (without power/energy).

        Returns
        -------
        list[LayerTiming]
            A list of LayerTiming objects with timing statistics only.

        """
        layer_stats: list[LayerTiming] = []

        for layer_name, times in self._timings.items():
            if not times:
                continue

            layer_timing = LayerTiming(
                name=layer_name,
                mean=mean(times),
                median=median(times),
                min=min(times),
                max=max(times),
                raw=times.copy(),
            )
            layer_stats.append(layer_timing)

        return layer_stats

    def correlate_with_tegrastats(
        self: Self,
        tegradata: TegraData,
        verbose: bool = False,
    ) -> list[JetsonLayerTiming]:
        """
        Correlate layer timestamps with tegrastats data to compute power and energy.

        For each layer, finds tegrastats samples that fall within the layer's
        execution time window and computes mean power and energy. If no samples
        are found, uses the last known power value.

        Parameters
        ----------
        tegradata : TegraData
            TegraData object containing parsed tegrastats data with timestamps and power measurements.
        verbose : bool, optional
            Whether to output verbose logging, by default False.

        Returns
        -------
        list[JetsonLayerTiming]
            A list of JetsonLayerTiming objects with timing, power, and energy metrics.

        """
        if verbose:
            LOG.info("Correlating layer timestamps with tegrastats data")

        # Build a sorted list of tegrastats samples with timestamps and power
        tegra_samples: list[tuple[float, float]] = []
        for sample in tegradata.data:
            # Each sample is a dict with string keys
            timestamp = float(sample["timestamp"])
            # Power values are stored as strings in the raw data, need to parse
            # We'll extract VDD_TOTAL from the power string
            # The power data is typically in the format we need to extract
            # For now, we'll use get_powerdraw to extract power from samples
            power_mw = 0.0
            # Check if we can extract power directly
            if "VDD_TOTAL" in sample:
                try:
                    power_mw = float(sample["VDD_TOTAL"].replace("mW", "").strip())
                except (ValueError, AttributeError):
                    power_mw = 0.0
            tegra_samples.append((timestamp, power_mw))
        
        tegra_samples.sort(key=lambda x: x[0])  # Sort by timestamp

        if not tegra_samples:
            LOG.warning("No tegrastats samples found")
            # Return basic LayerTiming wrapped as JetsonLayerTiming with zero power/energy
            return [
                JetsonLayerTiming(
                    name=lt.name,
                    mean=lt.mean,
                    median=lt.median,
                    min=lt.min,
                    max=lt.max,
                    raw=lt.raw,
                    power=0.0,
                    energy=0.0,
                )
                for lt in self.get_statistics()
            ]

        jetson_layer_stats: list[JetsonLayerTiming] = []
        last_known_power = tegra_samples[0][1] if tegra_samples else 0.0
        
        layers_with_data = 0
        layers_with_estimated = 0

        for layer_name, times in self._timings.items():
            if not times:
                continue

            timestamps = self._layer_timestamps[layer_name]
            
            # Collect power samples for this layer across all iterations
            layer_power_samples: list[float] = []
            layer_energy_samples: list[float] = []

            for (start_time, end_time), duration_ms in zip(timestamps, times):
                # Find tegrastats samples within this layer's time window
                matching_samples = [
                    power for ts, power in tegra_samples
                    if start_time <= ts <= end_time
                ]

                if matching_samples:
                    # Compute mean power for this iteration of the layer
                    mean_power = sum(matching_samples) / len(matching_samples)
                    layer_power_samples.append(mean_power)
                    last_known_power = mean_power
                else:
                    # No samples found, use last known power
                    layer_power_samples.append(last_known_power)
                
                # Compute energy: power (mW) * time (seconds)
                duration_seconds = duration_ms / 1000.0
                energy_mj = layer_power_samples[-1] * duration_seconds
                layer_energy_samples.append(energy_mj)

            # Track statistics
            if any(matching_samples for (start_time, end_time), _ in zip(timestamps, times)
                   for matching_samples in [[power for ts, power in tegra_samples if start_time <= ts <= end_time]]):
                layers_with_data += 1
            else:
                layers_with_estimated += 1

            # Compute statistics
            avg_power = sum(layer_power_samples) / len(layer_power_samples) if layer_power_samples else 0.0
            avg_energy = sum(layer_energy_samples) / len(layer_energy_samples) if layer_energy_samples else 0.0

            jetson_layer_timing = JetsonLayerTiming(
                name=layer_name,
                mean=mean(times),
                median=median(times),
                min=min(times),
                max=max(times),
                raw=times.copy(),
                power=avg_power,
                energy=avg_energy,
            )
            jetson_layer_stats.append(jetson_layer_timing)

        if verbose:
            LOG.info(f"Layers with direct tegrastats correlation: {layers_with_data}")
            LOG.info(f"Layers with estimated power (last known): {layers_with_estimated}")

        return jetson_layer_stats

    def reset(self: Self) -> None:
        """Reset all stored timings and timestamps."""
        self._timings.clear()
        self._layer_timestamps.clear()
        self._current_iteration_timings.clear()
        self._current_iteration_start_time = 0.0
        self._current_layer_start_time = 0.0


def profile_engine(
    engine: Path | str | TRTEngine,
    iterations: int = 10000,
    warmup_iterations: int = 10,
    tegra_interval: int = 5,
    dla_core: int | None = None,
    *,
    warmup: bool | None = None,
    verbose: bool | None = None,
) -> JetsonProfilerResult:
    """
    Profile a TensorRT engine layer-by-layer on a Jetson device.

    This function runs inference multiple times and collects per-layer execution
    times using TensorRT's IProfiler interface, along with power and energy metrics
    using tegrastats. It returns aggregated statistics (mean, median, min, max)
    for each layer across all iterations, plus per-layer power and energy consumption.

    Notes
    -----
    For best results, build the engine with profiling_verbosity set to DETAILED
    when calling build_engine. Otherwise, layer names may be numeric indices.
    
    The default iteration count is 10000 (higher than standard profiling) to ensure
    adequate tegrastats sampling coverage across all layers, especially fast-executing ones.

    Parameters
    ----------
    engine : Path | str | TRTEngine
        The engine to profile. Either a TRTEngine object or path to the engine file.
        If a path is given, then a TRTEngine will be created automatically.
    iterations : int, optional
        The number of profiling iterations to run, by default 10000.
        Higher iteration counts provide better coverage for per-layer power metrics.
    warmup_iterations : int, optional
        The number of warmup iterations to run before profiling, by default 10.
    tegra_interval : int, optional
        The interval in milliseconds between tegrastats samples, by default 5.
    dla_core : int, optional
        The DLA core to assign DLA layers of the engine to. Default is None.
        If None, any DLA layers will be assigned to DLA core 0.
    warmup : bool, optional
        Whether to do warmup iterations, by default None.
        If None, warmup will be set to True.
    verbose : bool, optional
        Whether to output additional information to stdout.
        Default None/False.

    Returns
    -------
    JetsonProfilerResult
        A dataclass containing per-layer timing/power/energy statistics,
        total execution time, overall power draw, and overall energy consumption.

    """
    if verbose:
        LOG.info("Starting Jetson engine profiling with per-layer power tracking")

    if warmup is None:
        warmup = True

    engine_loaded = False
    if isinstance(engine, (Path, str)):
        engine = TRTEngine(
            engine,
            dla_core=dla_core,
            warmup=False,
            verbose=verbose,
        )
        engine_loaded = True

    # issue warning if not built with detailed profiling
    import tensorrt as trt
    engine_verbosity = engine.engine.profiling_verbosity
    if engine_verbosity != trt.ProfilingVerbosity.DETAILED and verbose:
        LOG.warning(
            "Engine profiling verbosity is not DETAILED. Layer names may be numeric indices. "
            "Rebuild the engine with profiling_verbosity=trt.ProfilingVerbosity.DETAILED for best results.",
        )

    # attach JetsonLayerProfiler for per-layer power tracking
    profiler = JetsonLayerProfiler()
    engine.context.profiler = profiler

    # do warmup iterations
    # always do a single pass regardless of warmup_iterations
    profiler.start_iteration()
    engine.mock_execute(verbose=False)
    profiler.finalize_iteration()
    if warmup:
        for _ in range(warmup_iterations):
            profiler.start_iteration()
            engine.mock_execute(verbose=False)
            profiler.finalize_iteration()
    # report_layer_time is called by the context, so reset after warmup
    profiler.reset()

    if verbose:
        LOG.info(f"Running {iterations} profiling iterations with per-layer power monitoring")

    # pre-generate the false data
    false_data = engine.get_random_input(verbose=verbose)

    # create temp file location for tegrastats data
    temp_file = Path(Path.cwd()) / "temptegra_profile.txt"
    
    # store the start/stop times of each inference
    start_stop_times: list[tuple[float, float]] = []

    with TegraStats(temp_file, interval=tegra_interval):
        for idx in range(iterations):
            profiler.start_iteration()
            t0 = time.time()
            engine.mock_execute(false_data, verbose=False)
            t1 = time.time()
            profiler.finalize_iteration()
            start_stop_times.append((t0, t1))

            if verbose and (idx + 1) % 1000 == 0:
                LOG.info(f"Completed {idx + 1}/{iterations} iterations")

    # parse the tegrastats data first
    if verbose:
        LOG.info("Parsing tegrastats data")
    
    # Create TegraData object from the temp file
    with temp_file.open("r") as f:
        tegradata = TegraData(f)

    # delete the temp file
    if temp_file.exists():
        temp_file.unlink()

    # get layer statistics with power/energy correlation
    layer_stats = profiler.correlate_with_tegrastats(tegradata, verbose=verbose)

    if verbose:
        LOG.info(f"Profiling complete: {len(layer_stats)} layers profiled with power/energy metrics")

    # calculate total times per iteration
    total_times: list[float] = []
    for idx in range(iterations):
        iteration_total = sum(layer.raw[idx] for layer in layer_stats if idx < len(layer.raw))
        total_times.append(iteration_total)

    total_timing = LayerTiming(
        name="TOTAL",
        mean=sum(total_times) / len(total_times) if total_times else 0.0,
        median=sorted(total_times)[len(total_times) // 2] if total_times else 0.0,
        min=min(total_times) if total_times else 0.0,
        max=max(total_times) if total_times else 0.0,
        raw=total_times,
    )

    # filter the tegrastats data by actual times during execution
    tegradata.filter(start_stop_times)

    # get the overall power draw values using TegraData's property
    powerdraw_data: dict[str, JMetric] = tegradata.powerdraw
    power_raw = powerdraw_data["VDD_TOTAL"].raw

    # compute overall energy values per inference
    energy_data = [
        get_powerdraw(inf_data)["VDD_TOTAL"].mean * (inf_stop - inf_start)
        for (inf_start, inf_stop), inf_data in tegradata.filtered_entries
        if len(inf_data) > 0
    ]

    # create Metric objects for overall power and energy
    power_draw = Metric(power_raw)
    energy = Metric(energy_data)

    if verbose:
        LOG.info(f"Overall power draw: {power_draw}")
        LOG.info(f"Overall energy: {energy}")

    # if loaded here, delete engine
    if engine_loaded:
        del engine

    return JetsonProfilerResult(
        layers=layer_stats,
        total_time=total_timing,
        iterations=iterations,
        power_draw=power_draw,
        energy=energy,
    )
