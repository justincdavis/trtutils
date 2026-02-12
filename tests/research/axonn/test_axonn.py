# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for axonn data structures and cost functions."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add src to path to import without going through main trtutils package
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from trtutils.research.axonn._cost import (  # noqa: E402
    compute_dla_only_costs,
    compute_gpu_only_costs,
    compute_layer_energy,
    compute_layer_time,
    compute_total_energy,
    compute_total_time,
    create_dla_preferred_schedule,
    create_gpu_only_schedule,
    estimate_transition_cost,
)
from trtutils.research.axonn._types import (  # noqa: E402
    AxoNNConfig,
    Layer,
    LayerCost,
    ProcessorType,
    Schedule,
)


class TestLayerCost:
    """Tests for LayerCost dataclass."""

    def test_create_gpu_only(self) -> None:
        """Test creating a GPU-only layer cost."""
        cost = LayerCost(gpu_time_ms=1.0, gpu_energy_mj=0.5)
        assert cost.gpu_time_ms == 1.0
        assert cost.gpu_energy_mj == 0.5
        assert cost.dla_time_ms is None
        assert cost.dla_energy_mj is None

    def test_create_with_dla(self) -> None:
        """Test creating a layer cost with DLA values."""
        cost = LayerCost(
            gpu_time_ms=1.0,
            gpu_energy_mj=0.5,
            dla_time_ms=1.5,
            dla_energy_mj=0.3,
        )
        assert cost.gpu_time_ms == 1.0
        assert cost.gpu_energy_mj == 0.5
        assert cost.dla_time_ms == 1.5
        assert cost.dla_energy_mj == 0.3

    def test_str(self) -> None:
        """Test string representation."""
        cost = LayerCost(gpu_time_ms=1.0, gpu_energy_mj=0.5)
        assert "GPU:" in str(cost)
        assert "DLA: N/A" in str(cost)

        cost_dla = LayerCost(gpu_time_ms=1.0, gpu_energy_mj=0.5, dla_time_ms=1.5, dla_energy_mj=0.3)
        assert "DLA:" in str(cost_dla)
        assert "N/A" not in str(cost_dla)


class TestSchedule:
    """Tests for Schedule dataclass."""

    def test_empty_schedule(self) -> None:
        """Test creating an empty schedule."""
        schedule = Schedule()
        assert len(schedule.assignments) == 0
        assert schedule.num_transitions == 0
        assert schedule.total_time_ms == 0.0
        assert schedule.total_energy_mj == 0.0

    def test_num_transitions_property(self) -> None:
        """Test num_transitions is computed as a property."""
        schedule = Schedule()
        schedule.set_processor(0, ProcessorType.GPU)
        schedule.set_processor(1, ProcessorType.DLA)
        schedule.set_processor(2, ProcessorType.GPU)
        # GPU -> DLA -> GPU = 2 transitions
        assert schedule.num_transitions == 2

    def test_num_transitions_no_transitions(self) -> None:
        """Test num_transitions with no transitions."""
        schedule = Schedule()
        schedule.set_processor(0, ProcessorType.GPU)
        schedule.set_processor(1, ProcessorType.GPU)
        schedule.set_processor(2, ProcessorType.GPU)
        assert schedule.num_transitions == 0

    def test_num_transitions_single_layer(self) -> None:
        """Test num_transitions with single layer."""
        schedule = Schedule()
        schedule.set_processor(0, ProcessorType.GPU)
        assert schedule.num_transitions == 0

    def test_get_dla_layers(self) -> None:
        """Test getting DLA layer indices."""
        schedule = Schedule()
        schedule.set_processor(0, ProcessorType.GPU)
        schedule.set_processor(1, ProcessorType.DLA)
        schedule.set_processor(2, ProcessorType.DLA)
        schedule.set_processor(3, ProcessorType.GPU)
        assert sorted(schedule.get_dla_layers()) == [1, 2]

    def test_get_gpu_layers(self) -> None:
        """Test getting GPU layer indices."""
        schedule = Schedule()
        schedule.set_processor(0, ProcessorType.GPU)
        schedule.set_processor(1, ProcessorType.DLA)
        schedule.set_processor(2, ProcessorType.DLA)
        schedule.set_processor(3, ProcessorType.GPU)
        assert sorted(schedule.get_gpu_layers()) == [0, 3]


class TestEstimateTransitionCost:
    """Tests for estimate_transition_cost function."""

    def test_same_processor_no_cost(self) -> None:
        """Test that same processor transition has no cost."""
        layer = Layer(
            index=0,
            name="test",
            layer_type="Conv",
            output_tensor_size=1024,
            can_run_on_dla=True,
        )
        config = AxoNNConfig()
        time_ms, energy_mj = estimate_transition_cost(
            layer, ProcessorType.GPU, ProcessorType.GPU, config
        )
        assert time_ms == 0.0
        assert energy_mj == 0.0

    def test_different_processor_has_cost(self) -> None:
        """Test that different processor transition has cost."""
        layer = Layer(
            index=0,
            name="test",
            layer_type="Conv",
            output_tensor_size=1024 * 1024,  # 1 MB
            can_run_on_dla=True,
        )
        config = AxoNNConfig()
        time_ms, energy_mj = estimate_transition_cost(
            layer, ProcessorType.GPU, ProcessorType.DLA, config
        )
        assert time_ms > 0.0
        assert energy_mj > 0.0

    def test_returns_tuple(self) -> None:
        """Test that function returns a tuple."""
        layer = Layer(
            index=0,
            name="test",
            layer_type="Conv",
            output_tensor_size=1024,
            can_run_on_dla=True,
        )
        config = AxoNNConfig()
        result = estimate_transition_cost(layer, ProcessorType.GPU, ProcessorType.DLA, config)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestComputeLayerCosts:
    """Tests for compute_layer_time and compute_layer_energy."""

    def test_compute_layer_time_gpu(self) -> None:
        """Test computing layer time on GPU."""
        cost = LayerCost(gpu_time_ms=1.0, gpu_energy_mj=0.5)
        assert compute_layer_time(cost, ProcessorType.GPU) == 1.0

    def test_compute_layer_time_dla(self) -> None:
        """Test computing layer time on DLA."""
        cost = LayerCost(gpu_time_ms=1.0, gpu_energy_mj=0.5, dla_time_ms=1.5, dla_energy_mj=0.3)
        assert compute_layer_time(cost, ProcessorType.DLA) == 1.5

    def test_compute_layer_time_dla_not_available(self) -> None:
        """Test error when DLA time not available."""
        cost = LayerCost(gpu_time_ms=1.0, gpu_energy_mj=0.5)
        with pytest.raises(ValueError, match="not DLA-compatible"):
            compute_layer_time(cost, ProcessorType.DLA, layer_idx=0)

    def test_compute_layer_energy_gpu(self) -> None:
        """Test computing layer energy on GPU."""
        cost = LayerCost(gpu_time_ms=1.0, gpu_energy_mj=0.5)
        assert compute_layer_energy(cost, ProcessorType.GPU) == 0.5

    def test_compute_layer_energy_dla(self) -> None:
        """Test computing layer energy on DLA."""
        cost = LayerCost(gpu_time_ms=1.0, gpu_energy_mj=0.5, dla_time_ms=1.5, dla_energy_mj=0.3)
        assert compute_layer_energy(cost, ProcessorType.DLA) == 0.3

    def test_compute_layer_energy_dla_not_available(self) -> None:
        """Test error when DLA energy not available."""
        cost = LayerCost(gpu_time_ms=1.0, gpu_energy_mj=0.5)
        with pytest.raises(ValueError, match="not DLA-compatible"):
            compute_layer_energy(cost, ProcessorType.DLA, layer_idx=0)


class TestComputeTotalCosts:
    """Tests for compute_total_time and compute_total_energy."""

    @pytest.fixture
    def layers_and_costs(
        self,
    ) -> tuple[list[Layer], dict[int, LayerCost], AxoNNConfig]:
        """Create test layers and costs."""
        layers = [
            Layer(
                index=0,
                name="layer0",
                layer_type="Conv",
                output_tensor_size=1024,
                can_run_on_dla=True,
            ),
            Layer(
                index=1,
                name="layer1",
                layer_type="Pool",
                output_tensor_size=512,
                can_run_on_dla=True,
            ),
            Layer(
                index=2,
                name="layer2",
                layer_type="Conv",
                output_tensor_size=256,
                can_run_on_dla=False,
            ),
        ]
        costs = {
            0: LayerCost(gpu_time_ms=1.0, gpu_energy_mj=0.5, dla_time_ms=1.5, dla_energy_mj=0.3),
            1: LayerCost(gpu_time_ms=0.5, gpu_energy_mj=0.3, dla_time_ms=0.8, dla_energy_mj=0.2),
            2: LayerCost(gpu_time_ms=2.0, gpu_energy_mj=1.0),  # GPU only
        }
        config = AxoNNConfig()
        return layers, costs, config

    def test_compute_total_time_all_gpu(
        self,
        layers_and_costs: tuple[list[Layer], dict[int, LayerCost], AxoNNConfig],
    ) -> None:
        """Test computing total time with all GPU schedule."""
        layers, costs, config = layers_and_costs
        schedule = create_gpu_only_schedule(layers)
        total_time = compute_total_time(layers, costs, schedule, config)
        # 1.0 + 0.5 + 2.0 = 3.5
        assert total_time == pytest.approx(3.5)

    def test_compute_total_energy_all_gpu(
        self,
        layers_and_costs: tuple[list[Layer], dict[int, LayerCost], AxoNNConfig],
    ) -> None:
        """Test computing total energy with all GPU schedule."""
        layers, costs, config = layers_and_costs
        schedule = create_gpu_only_schedule(layers)
        total_energy = compute_total_energy(layers, costs, schedule, config)
        # 0.5 + 0.3 + 1.0 = 1.8
        assert total_energy == pytest.approx(1.8)


class TestComputeGpuOnlyCosts:
    """Tests for compute_gpu_only_costs."""

    def test_compute_gpu_only_costs(self) -> None:
        """Test computing GPU-only costs."""
        costs = {
            0: LayerCost(gpu_time_ms=1.0, gpu_energy_mj=0.5),
            1: LayerCost(gpu_time_ms=2.0, gpu_energy_mj=1.0),
        }
        total_time, total_energy = compute_gpu_only_costs(costs)
        assert total_time == pytest.approx(3.0)
        assert total_energy == pytest.approx(1.5)


class TestComputeDlaOnlyCosts:
    """Tests for compute_dla_only_costs."""

    def test_compute_dla_only_costs_mixed(self) -> None:
        """Test computing DLA-preferred costs with mixed layers."""
        costs = {
            0: LayerCost(gpu_time_ms=1.0, gpu_energy_mj=0.5, dla_time_ms=1.5, dla_energy_mj=0.3),
            1: LayerCost(gpu_time_ms=2.0, gpu_energy_mj=1.0),  # GPU only
        }
        total_time, total_energy = compute_dla_only_costs(costs)
        # Layer 0 on DLA: 1.5, Layer 1 on GPU: 2.0
        assert total_time == pytest.approx(3.5)
        # Layer 0 on DLA: 0.3, Layer 1 on GPU: 1.0
        assert total_energy == pytest.approx(1.3)


class TestCreateSchedules:
    """Tests for create_gpu_only_schedule and create_dla_preferred_schedule."""

    @pytest.fixture
    def layers(self) -> list[Layer]:
        """Create test layers."""
        return [
            Layer(
                index=0,
                name="layer0",
                layer_type="Conv",
                output_tensor_size=1024,
                can_run_on_dla=True,
            ),
            Layer(
                index=1,
                name="layer1",
                layer_type="Pool",
                output_tensor_size=512,
                can_run_on_dla=False,
            ),
            Layer(
                index=2,
                name="layer2",
                layer_type="Conv",
                output_tensor_size=256,
                can_run_on_dla=True,
            ),
        ]

    def test_create_gpu_only_schedule(self, layers: list[Layer]) -> None:
        """Test creating GPU-only schedule."""
        schedule = create_gpu_only_schedule(layers)
        assert all(p == ProcessorType.GPU for p in schedule.assignments.values())
        assert schedule.num_transitions == 0

    def test_create_dla_preferred_schedule(self, layers: list[Layer]) -> None:
        """Test creating DLA-preferred schedule."""
        schedule = create_dla_preferred_schedule(layers)
        assert schedule.assignments[0] == ProcessorType.DLA
        assert schedule.assignments[1] == ProcessorType.GPU  # Not DLA-compatible
        assert schedule.assignments[2] == ProcessorType.DLA
        # DLA -> GPU -> DLA = 2 transitions
        assert schedule.num_transitions == 2
