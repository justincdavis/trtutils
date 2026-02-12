# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for haxconn data structures, grouping, contention model, and cost functions."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add src to path to import without going through main trtutils package
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from trtutils.research.haxconn._contention import (  # noqa: E402
    compute_contention_interval,
    compute_contention_slowdown,
)
from trtutils.research.haxconn._cost import (  # noqa: E402
    compute_dnn_time,
    compute_dnn_timing,
    compute_group_time,
    compute_max_latency_objective,
    compute_throughput_objective,
    create_naive_schedule,
    estimate_transition_cost,
)
from trtutils.research.haxconn._grouping import (  # noqa: E402
    assign_groups_to_layers,
    identify_layer_groups,
)
from trtutils.research.haxconn._types import (  # noqa: E402
    DNNSchedule,
    HaxconnConfig,
    Layer,
    LayerGroup,
    LayerGroupCost,
    MultiSchedule,
    Objective,
    ProcessorType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _layer(
    index: int,
    name: str,
    layer_type: str,
    output_tensor_size: int,
    *,
    dla: bool,
    input_tensor_size: int = 0,
) -> Layer:
    """Create a Layer with keyword-only DLA flag."""
    return Layer(
        index=index,
        name=name,
        layer_type=layer_type,
        output_tensor_size=output_tensor_size,
        can_run_on_dla=dla,
        input_tensor_size=input_tensor_size,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_layers() -> list[Layer]:
    """Four layers: DLA, DLA, GPU-only, DLA."""
    return [
        _layer(0, "conv1", "CONVOLUTION", 1000, dla=True, input_tensor_size=500),
        _layer(1, "relu1", "ACTIVATION", 1000, dla=True, input_tensor_size=1000),
        _layer(2, "reshape", "SHUFFLE", 500, dla=False, input_tensor_size=1000),
        _layer(3, "conv2", "CONVOLUTION", 2000, dla=True, input_tensor_size=500),
    ]


@pytest.fixture
def config() -> HaxconnConfig:
    """Default HaX-CoNN config."""
    return HaxconnConfig()


@pytest.fixture
def simple_groups(simple_layers: list[Layer]) -> list[LayerGroup]:
    """Derive groups from simple_layers."""
    return identify_layer_groups(simple_layers, dnn_id=0)


@pytest.fixture
def simple_costs() -> dict[int, LayerGroupCost]:
    """Cost data for 3 groups (group ids 0, 1, 2)."""
    return {
        0: LayerGroupCost(
            gpu_time_ms=1.0,
            gpu_energy_mj=0.5,
            gpu_mem_throughput_mbps=1000.0,
            dla_time_ms=1.5,
            dla_energy_mj=0.3,
            dla_mem_throughput_mbps=800.0,
        ),
        1: LayerGroupCost(
            gpu_time_ms=0.5,
            gpu_energy_mj=0.2,
            gpu_mem_throughput_mbps=500.0,
        ),
        2: LayerGroupCost(
            gpu_time_ms=2.0,
            gpu_energy_mj=1.0,
            gpu_mem_throughput_mbps=2000.0,
            dla_time_ms=2.5,
            dla_energy_mj=0.6,
            dla_mem_throughput_mbps=1500.0,
        ),
    }


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class TestLayerGroupCost:
    """Tests for LayerGroupCost dataclass."""

    def test_create_gpu_only(self) -> None:
        """Test creating a GPU-only group cost."""
        cost = LayerGroupCost(gpu_time_ms=1.0, gpu_energy_mj=0.5)
        assert cost.gpu_time_ms == 1.0
        assert cost.gpu_energy_mj == 0.5
        assert cost.gpu_mem_throughput_mbps == 0.0
        assert cost.dla_time_ms is None
        assert cost.dla_energy_mj is None
        assert cost.dla_mem_throughput_mbps is None

    def test_create_with_dla(self) -> None:
        """Test creating a group cost with DLA values."""
        cost = LayerGroupCost(
            gpu_time_ms=1.0,
            gpu_energy_mj=0.5,
            gpu_mem_throughput_mbps=1000.0,
            dla_time_ms=1.5,
            dla_energy_mj=0.3,
            dla_mem_throughput_mbps=800.0,
        )
        assert cost.dla_time_ms == 1.5
        assert cost.dla_energy_mj == 0.3
        assert cost.dla_mem_throughput_mbps == 800.0

    def test_str_gpu_only(self) -> None:
        """Test string representation without DLA."""
        cost = LayerGroupCost(gpu_time_ms=1.0, gpu_energy_mj=0.5)
        s = str(cost)
        assert "GPU:" in s
        assert "DLA: N/A" in s

    def test_str_with_dla(self) -> None:
        """Test string representation with DLA."""
        cost = LayerGroupCost(
            gpu_time_ms=1.0,
            gpu_energy_mj=0.5,
            dla_time_ms=1.5,
            dla_energy_mj=0.3,
        )
        s = str(cost)
        assert "DLA:" in s
        assert "N/A" not in s


class TestDNNSchedule:
    """Tests for DNNSchedule dataclass."""

    def test_empty_schedule(self) -> None:
        """Test creating an empty schedule."""
        schedule = DNNSchedule(dnn_id=0)
        assert len(schedule.assignments) == 0
        assert schedule.total_time_ms == 0.0

    def test_set_and_get_processor(self) -> None:
        """Test setting and getting processor assignments."""
        schedule = DNNSchedule(dnn_id=0)
        schedule.set_processor(0, ProcessorType.GPU)
        schedule.set_processor(1, ProcessorType.DLA)
        assert schedule.get_processor(0) == ProcessorType.GPU
        assert schedule.get_processor(1) == ProcessorType.DLA

    def test_get_dla_and_gpu_groups(self) -> None:
        """Test retrieving DLA and GPU group lists."""
        schedule = DNNSchedule(dnn_id=0)
        schedule.set_processor(0, ProcessorType.GPU)
        schedule.set_processor(1, ProcessorType.DLA)
        schedule.set_processor(2, ProcessorType.DLA)
        assert sorted(schedule.get_dla_groups()) == [1, 2]
        assert schedule.get_gpu_groups() == [0]


class TestMultiSchedule:
    """Tests for MultiSchedule dataclass."""

    def test_empty(self) -> None:
        """Test empty multi-schedule has 0 DNNs."""
        ms = MultiSchedule()
        assert ms.num_dnns == 0

    def test_get_schedule(self) -> None:
        """Test get_schedule returns correct DNN schedule."""
        s0 = DNNSchedule(dnn_id=0)
        s1 = DNNSchedule(dnn_id=1)
        ms = MultiSchedule(dnn_schedules=[s0, s1])
        assert ms.get_schedule(0) is s0
        assert ms.get_schedule(1) is s1

    def test_get_schedule_missing(self) -> None:
        """Test get_schedule raises KeyError for missing DNN."""
        ms = MultiSchedule()
        with pytest.raises(KeyError, match="No schedule found"):
            ms.get_schedule(99)


class TestHaxconnConfig:
    """Tests for HaxconnConfig dataclass."""

    def test_defaults(self) -> None:
        """Test default config values."""
        cfg = HaxconnConfig()
        assert cfg.objective == Objective.MAX_THROUGHPUT
        assert cfg.pccs_alpha == 0.5
        assert cfg.pccs_beta == 1.5
        assert cfg.max_bandwidth_mbps == 25600.0

    def test_str(self) -> None:
        """Test string representation."""
        cfg = HaxconnConfig()
        s = str(cfg)
        assert "MAX_THROUGHPUT" in s
        assert "alpha=" in s


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


class TestLayerGrouping:
    """Tests for identify_layer_groups and assign_groups_to_layers."""

    def test_groups_split_on_dla_boundary(self, simple_layers: list[Layer]) -> None:
        """Test groups split when DLA compatibility changes."""
        groups = identify_layer_groups(simple_layers, dnn_id=0)
        # DLA(0,1), GPU(2), DLA(3) => 3 groups
        assert len(groups) == 3

    def test_first_group_is_dla(self, simple_groups: list[LayerGroup]) -> None:
        """Test first group is DLA-compatible with layers 0,1."""
        assert simple_groups[0].can_run_on_dla is True
        assert simple_groups[0].layer_indices == [0, 1]

    def test_second_group_is_gpu_only(self, simple_groups: list[LayerGroup]) -> None:
        """Test second group is GPU-only with layer 2."""
        assert simple_groups[1].can_run_on_dla is False
        assert simple_groups[1].layer_indices == [2]

    def test_third_group_is_dla(self, simple_groups: list[LayerGroup]) -> None:
        """Test third group is DLA-compatible with layer 3."""
        assert simple_groups[2].can_run_on_dla is True
        assert simple_groups[2].layer_indices == [3]

    def test_group_ids_sequential(self, simple_groups: list[LayerGroup]) -> None:
        """Test group IDs are assigned sequentially starting from 0."""
        assert [g.group_id for g in simple_groups] == [0, 1, 2]

    def test_dnn_id_propagated(self, simple_groups: list[LayerGroup]) -> None:
        """Test dnn_id is set on all groups."""
        assert all(g.dnn_id == 0 for g in simple_groups)

    def test_aggregate_tensor_sizes(self, simple_groups: list[LayerGroup]) -> None:
        """Test output tensor sizes are aggregated across layers in group."""
        # Group 0 has layers 0,1: output_sizes 1000+1000 = 2000
        assert simple_groups[0].total_output_tensor_size == 2000

    def test_num_layers_property(self, simple_groups: list[LayerGroup]) -> None:
        """Test num_layers property returns correct counts."""
        assert simple_groups[0].num_layers == 2
        assert simple_groups[1].num_layers == 1
        assert simple_groups[2].num_layers == 1

    def test_empty_layers(self) -> None:
        """Test grouping empty layer list returns empty."""
        groups = identify_layer_groups([], dnn_id=0)
        assert groups == []

    def test_single_layer(self) -> None:
        """Test grouping a single layer produces one group."""
        layers = [_layer(0, "conv", "CONVOLUTION", 100, dla=True, input_tensor_size=50)]
        groups = identify_layer_groups(layers, dnn_id=0)
        assert len(groups) == 1
        assert groups[0].layer_indices == [0]

    def test_all_same_dla_compat(self) -> None:
        """Test all-DLA layers produce a single group."""
        layers = [
            _layer(0, "a", "CONV", 100, dla=True, input_tensor_size=50),
            _layer(1, "b", "CONV", 100, dla=True, input_tensor_size=50),
            _layer(2, "c", "CONV", 100, dla=True, input_tensor_size=50),
        ]
        groups = identify_layer_groups(layers, dnn_id=0)
        assert len(groups) == 1
        assert groups[0].layer_indices == [0, 1, 2]

    def test_assign_groups_to_layers(
        self,
        simple_layers: list[Layer],
        simple_groups: list[LayerGroup],
    ) -> None:
        """Test assign_groups_to_layers sets group_id on each layer."""
        updated = assign_groups_to_layers(simple_layers, simple_groups)
        assert updated[0].group_id == 0
        assert updated[1].group_id == 0
        assert updated[2].group_id == 1
        assert updated[3].group_id == 2


# ---------------------------------------------------------------------------
# Contention model (PCCS)
# ---------------------------------------------------------------------------


class TestContentionSlowdown:
    """Tests for compute_contention_slowdown."""

    def test_zero_throughput_no_slowdown(self, config: HaxconnConfig) -> None:
        """Test zero throughput yields no slowdown."""
        assert compute_contention_slowdown(0.0, 0.0, config) == 1.0

    def test_positive_throughput_gives_slowdown_gt_1(self, config: HaxconnConfig) -> None:
        """Test positive throughput produces slowdown > 1."""
        slowdown = compute_contention_slowdown(5000.0, 5000.0, config)
        assert slowdown > 1.0

    def test_higher_throughput_gives_more_slowdown(self, config: HaxconnConfig) -> None:
        """Test that higher combined throughput yields more slowdown."""
        low = compute_contention_slowdown(1000.0, 1000.0, config)
        high = compute_contention_slowdown(10000.0, 10000.0, config)
        assert high > low

    def test_zero_bandwidth_returns_1(self) -> None:
        """Test zero max bandwidth disables contention."""
        cfg = HaxconnConfig(max_bandwidth_mbps=0.0)
        assert compute_contention_slowdown(5000.0, 5000.0, cfg) == 1.0

    def test_symmetric(self, config: HaxconnConfig) -> None:
        """Test contention is symmetric in throughput_a and throughput_b."""
        s1 = compute_contention_slowdown(3000.0, 5000.0, config)
        s2 = compute_contention_slowdown(5000.0, 3000.0, config)
        assert s1 == pytest.approx(s2)


class TestContentionInterval:
    """Tests for compute_contention_interval."""

    def test_full_overlap(self) -> None:
        """Test identical windows yield full overlap."""
        assert compute_contention_interval(0.0, 5.0, 0.0, 5.0) == pytest.approx(5.0)

    def test_partial_overlap(self) -> None:
        """Test partially overlapping windows."""
        assert compute_contention_interval(0.0, 5.0, 3.0, 8.0) == pytest.approx(2.0)

    def test_no_overlap(self) -> None:
        """Test non-overlapping windows yield 0."""
        assert compute_contention_interval(0.0, 3.0, 5.0, 8.0) == pytest.approx(0.0)

    def test_touching_no_overlap(self) -> None:
        """Test touching (end == start) windows yield 0."""
        assert compute_contention_interval(0.0, 3.0, 3.0, 6.0) == pytest.approx(0.0)

    def test_contained(self) -> None:
        """Test one window fully contained in another."""
        assert compute_contention_interval(0.0, 10.0, 2.0, 5.0) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Cost functions
# ---------------------------------------------------------------------------


class TestEstimateTransitionCost:
    """Tests for estimate_transition_cost."""

    def test_same_processor_no_cost(self, config: HaxconnConfig) -> None:
        """Test no cost for same-processor transition."""
        group = LayerGroup(group_id=0, dnn_id=0, total_output_tensor_size=1024)
        cost = estimate_transition_cost(group, ProcessorType.GPU, ProcessorType.GPU, config)
        assert cost == 0.0

    def test_different_processor_has_cost(self, config: HaxconnConfig) -> None:
        """Test positive cost for GPU-DLA transition."""
        group = LayerGroup(group_id=0, dnn_id=0, total_output_tensor_size=1024 * 1024)
        cost = estimate_transition_cost(group, ProcessorType.GPU, ProcessorType.DLA, config)
        assert cost > 0.0

    def test_larger_tensor_more_cost(self, config: HaxconnConfig) -> None:
        """Test larger tensors incur higher transition cost."""
        small = LayerGroup(group_id=0, dnn_id=0, total_output_tensor_size=1024)
        big = LayerGroup(group_id=1, dnn_id=0, total_output_tensor_size=1024 * 1024 * 10)
        c_small = estimate_transition_cost(small, ProcessorType.GPU, ProcessorType.DLA, config)
        c_big = estimate_transition_cost(big, ProcessorType.GPU, ProcessorType.DLA, config)
        assert c_big > c_small


class TestComputeGroupTime:
    """Tests for compute_group_time."""

    def test_gpu_time(self) -> None:
        """Test GPU time is returned for GPU processor."""
        cost = LayerGroupCost(gpu_time_ms=1.0, gpu_energy_mj=0.5)
        assert compute_group_time(cost, ProcessorType.GPU) == 1.0

    def test_dla_time(self) -> None:
        """Test DLA time is returned for DLA processor."""
        cost = LayerGroupCost(gpu_time_ms=1.0, gpu_energy_mj=0.5, dla_time_ms=1.5, dla_energy_mj=0.3)
        assert compute_group_time(cost, ProcessorType.DLA) == 1.5

    def test_dla_time_not_available(self) -> None:
        """Test error when DLA time not available."""
        cost = LayerGroupCost(gpu_time_ms=1.0, gpu_energy_mj=0.5)
        with pytest.raises(ValueError, match="not DLA-compatible"):
            compute_group_time(cost, ProcessorType.DLA, group_id=0)


class TestComputeDnnTime:
    """Tests for compute_dnn_time."""

    def test_all_gpu_no_transitions(
        self,
        simple_groups: list[LayerGroup],
        simple_costs: dict[int, LayerGroupCost],
        config: HaxconnConfig,
    ) -> None:
        """Test all-GPU schedule sums group times without transitions."""
        schedule = DNNSchedule(dnn_id=0)
        for g in simple_groups:
            schedule.set_processor(g.group_id, ProcessorType.GPU)
        total = compute_dnn_time(simple_groups, simple_costs, schedule, config)
        # 1.0 + 0.5 + 2.0 = 3.5
        assert total == pytest.approx(3.5)

    def test_mixed_schedule_includes_transition(
        self,
        simple_groups: list[LayerGroup],
        simple_costs: dict[int, LayerGroupCost],
        config: HaxconnConfig,
    ) -> None:
        """Test mixed GPU/DLA schedule includes transition costs."""
        schedule = DNNSchedule(dnn_id=0)
        schedule.set_processor(0, ProcessorType.DLA)
        schedule.set_processor(1, ProcessorType.GPU)
        schedule.set_processor(2, ProcessorType.DLA)
        total = compute_dnn_time(simple_groups, simple_costs, schedule, config)
        # 1.5 (DLA) + transition + 0.5 (GPU) + transition + 2.5 (DLA) > 4.5
        assert total > 4.5


class TestComputeDnnTiming:
    """Tests for compute_dnn_timing populating start/end times."""

    def test_populates_start_end_times(
        self,
        simple_groups: list[LayerGroup],
        simple_costs: dict[int, LayerGroupCost],
        config: HaxconnConfig,
    ) -> None:
        """Test that start and end times are populated for all groups."""
        schedule = DNNSchedule(dnn_id=0)
        for g in simple_groups:
            schedule.set_processor(g.group_id, ProcessorType.GPU)
        result = compute_dnn_timing(simple_groups, simple_costs, schedule, config)
        for g in simple_groups:
            assert g.group_id in result.group_start_times
            assert g.group_id in result.group_end_times
            assert result.group_end_times[g.group_id] > result.group_start_times[g.group_id]

    def test_groups_are_sequential(
        self,
        simple_groups: list[LayerGroup],
        simple_costs: dict[int, LayerGroupCost],
        config: HaxconnConfig,
    ) -> None:
        """Test that group end time equals next group start time."""
        schedule = DNNSchedule(dnn_id=0)
        for g in simple_groups:
            schedule.set_processor(g.group_id, ProcessorType.GPU)
        result = compute_dnn_timing(simple_groups, simple_costs, schedule, config)
        sorted_ids = sorted(result.group_start_times.keys())
        for i in range(len(sorted_ids) - 1):
            assert result.group_end_times[sorted_ids[i]] == pytest.approx(
                result.group_start_times[sorted_ids[i + 1]]
            )

    def test_total_time_set(
        self,
        simple_groups: list[LayerGroup],
        simple_costs: dict[int, LayerGroupCost],
        config: HaxconnConfig,
    ) -> None:
        """Test total_time_ms is set on the schedule."""
        schedule = DNNSchedule(dnn_id=0)
        for g in simple_groups:
            schedule.set_processor(g.group_id, ProcessorType.GPU)
        result = compute_dnn_timing(simple_groups, simple_costs, schedule, config)
        assert result.total_time_ms == pytest.approx(3.5)


class TestObjectives:
    """Tests for throughput and max-latency objectives."""

    def test_throughput_sums_times(self) -> None:
        """Test throughput objective sums all DNN latencies."""
        s0 = DNNSchedule(dnn_id=0, total_time_ms=3.0)
        s1 = DNNSchedule(dnn_id=1, total_time_ms=5.0)
        ms = MultiSchedule(dnn_schedules=[s0, s1])
        assert compute_throughput_objective(ms) == pytest.approx(8.0)

    def test_max_latency_picks_max(self) -> None:
        """Test max-latency objective picks the maximum."""
        s0 = DNNSchedule(dnn_id=0, total_time_ms=3.0)
        s1 = DNNSchedule(dnn_id=1, total_time_ms=5.0)
        ms = MultiSchedule(dnn_schedules=[s0, s1])
        assert compute_max_latency_objective(ms) == pytest.approx(5.0)

    def test_max_latency_empty(self) -> None:
        """Test max-latency returns 0 for empty schedule."""
        ms = MultiSchedule()
        assert compute_max_latency_objective(ms) == 0.0


class TestCreateNaiveSchedule:
    """Tests for create_naive_schedule."""

    def test_all_gpu_assignments(
        self,
        simple_groups: list[LayerGroup],
        simple_costs: dict[int, LayerGroupCost],
        config: HaxconnConfig,
    ) -> None:
        """Test naive schedule assigns all groups to GPU."""
        ms = create_naive_schedule([simple_groups], [simple_costs], config)
        assert ms.num_dnns == 1
        sched = ms.get_schedule(0)
        for g in simple_groups:
            assert sched.get_processor(g.group_id) == ProcessorType.GPU

    def test_objectives_populated(
        self,
        simple_groups: list[LayerGroup],
        simple_costs: dict[int, LayerGroupCost],
        config: HaxconnConfig,
    ) -> None:
        """Test objectives are computed and positive."""
        ms = create_naive_schedule([simple_groups], [simple_costs], config)
        assert ms.throughput_objective > 0
        assert ms.max_latency_objective > 0

    def test_multi_dnn(self, config: HaxconnConfig) -> None:
        """Test naive schedule with two DNNs."""
        groups_a = [LayerGroup(group_id=0, dnn_id=0, layer_indices=[0])]
        groups_b = [LayerGroup(group_id=0, dnn_id=1, layer_indices=[0])]
        costs_a = {0: LayerGroupCost(gpu_time_ms=1.0, gpu_energy_mj=0.5)}
        costs_b = {0: LayerGroupCost(gpu_time_ms=2.0, gpu_energy_mj=1.0)}
        ms = create_naive_schedule([groups_a, groups_b], [costs_a, costs_b], config)
        assert ms.num_dnns == 2
        assert ms.throughput_objective == pytest.approx(3.0)
        assert ms.max_latency_objective == pytest.approx(2.0)
