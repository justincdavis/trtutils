# AxoNN Methodology Summary

## Overview

AxoNN finds optimal layer-to-accelerator mappings for neural network inference on heterogeneous SoCs (e.g., NVIDIA Xavier with GPU + DLA) by solving a constrained optimization problem.

**Goal**: Minimize execution time while staying under an Energy Consumption Target (ECT).

---

## Core Data Structures

### Layer Set `L`

```python
# Each layer i in the network
layers: list[Layer]  # 0 <= i < len(L)

class Layer:
    index: int
    type: str  # "conv", "pool", "relu", etc.
    output_tensor_size: int  # bytes
    can_run_on_dla: bool
    can_be_transition_point: bool
```

### Processor Set `P`

```python
# Available accelerators
processors: list[Processor]  # e.g., [GPU, DLA]

class Processor:
    name: str
    is_performance_oriented: bool  # GPU = True, DLA = False
    is_energy_efficient: bool      # DLA = True, GPU = False
```

### Schedule `S`

```python
# Maps each layer index to a processor
schedule: dict[int, Processor]  # S(i) = p_j

def get_schedule(layer_idx: int) -> Processor:
    """Returns which processor runs layer i."""
    return schedule[layer_idx]
```

---

## Cost Models

### 1. Layer Execution Time & Energy

For each layer on each processor, profile:

```python
def exec_time(layer: Layer, processor: Processor) -> float:
    """T(l_i, p_j): Execution time of layer i on processor j (ms)."""
    return profiled_times[(layer.index, processor.name)]

def energy(layer: Layer, processor: Processor) -> float:
    """E(l_i, p_j): Energy consumption of layer i on processor j (mJ)."""
    return profiled_energy[(layer.index, processor.name)]
```

### 2. Transition Costs

When execution switches between accelerators, there's overhead from cache flushes and memory operations.

```python
def output_transition_cost(
    layer: Layer,
    from_proc: Processor,
    to_proc: Processor
) -> float:
    """
    T_out(l_i, p_from, p_to): Cost of flushing layer i's output
    from p_from's cache back to shared memory.
    
    Depends on: output tensor size, layer type, source processor
    """
    if from_proc == to_proc:
        return 0.0
    return transition_model(layer.output_tensor_size, from_proc)

def input_transition_cost(
    layer: Layer,
    from_proc: Processor,
    to_proc: Processor
) -> float:
    """
    T_in(l_i, p_from, p_to): Cold cache miss cost when p_to
    starts reading layer i's input from shared memory.
    
    Depends on: input tensor size, layer type, destination processor
    """
    if from_proc == to_proc:
        return 0.0
    return transition_model(layer.input_tensor_size, to_proc)
```

### 3. Pipeline Breaking Cost

Some accelerators (like DLA) have internal hardware pipelines. Breaking them incurs penalties.

```python
def pipeline_cost(layer: Layer, schedule: Schedule) -> float:
    """
    P(l_i, S(l_i)): Extra cost if layer breaks an internal pipeline.
    
    Returns 0 if previous layer used the same sub-unit.
    Returns penalty if sub-units differ (e.g., conv -> pool on DLA).
    """
    if layer.index == 0:
        return 0.0
    
    curr_proc = schedule[layer.index]
    prev_proc = schedule[layer.index - 1]
    
    if curr_proc != prev_proc:
        return 0.0  # Transition handles this
    
    # Check if internal sub-units match
    curr_subunit = get_subunit(layer)
    prev_subunit = get_subunit(layers[layer.index - 1])
    
    if curr_subunit == prev_subunit:
        return 0.0
    return pipeline_break_penalty(layer, curr_subunit)
```

---

## Total Cost Functions

### Total Execution Time

```python
def total_time(
    layers: list[Layer],
    processors: list[Processor],
    schedule: Schedule
) -> float:
    """
    T_total = sum over all layers of:
        - Layer execution time
        - Output transition cost (if transitioning)
        - Input transition cost (if transitioning)  
        - Pipeline breaking cost
    """
    total = 0.0
    
    for i, layer in enumerate(layers):
        proc = schedule[i]
        
        # Base execution time
        total += exec_time(layer, proc)
        
        # Pipeline cost
        total += pipeline_cost(layer, schedule)
        
        # Transition costs (only if switching processors)
        if i > 0 and schedule[i] != schedule[i - 1]:
            prev_proc = schedule[i - 1]
            # Output cost from previous processor
            total += output_transition_cost(layers[i-1], prev_proc, proc)
            # Input cost to current processor
            total += input_transition_cost(layer, prev_proc, proc)
    
    return total
```

### Total Energy

```python
def total_energy(
    layers: list[Layer],
    processors: list[Processor],
    schedule: Schedule
) -> float:
    """
    E_total = sum over all layers of:
        - Layer energy consumption
        - Transition energy costs
    """
    total = 0.0
    
    for i, layer in enumerate(layers):
        proc = schedule[i]
        total += energy(layer, proc)
        
        # Transition energy (if switching)
        if i > 0 and schedule[i] != schedule[i - 1]:
            prev_proc = schedule[i - 1]
            total += transition_energy(layers[i-1], prev_proc, proc)
            total += transition_energy(layer, prev_proc, proc)
    
    return total
```

---

## Optimization Problem

### Formulation

```python
# Objective: Minimize execution time
# Constraint: Energy must be under target ECT

minimize:    total_time(L, P, S)
subject_to:  total_energy(L, P, S) <= ECT
```

### Transition Indicator

```python
def is_transition(schedule: Schedule, layer_idx: int) -> bool:
    """t_i: Boolean indicating if a transition occurs after layer i."""
    if layer_idx >= len(schedule) - 1:
        return False
    return schedule[layer_idx] != schedule[layer_idx + 1]

def count_transitions(schedule: Schedule) -> int:
    """Total number of inter-DSA transitions."""
    return sum(1 for i in range(len(schedule) - 1) if is_transition(schedule, i))
```

### Additional Constraint (Optional)

```python
# Limit number of transitions for faster solving
subject_to:  count_transitions(S) <= max_transitions
```

---

## Profiling Methodology

### Step 1: Check Layer Compatibility

```python
# Using TensorRT API
def check_layer_compatibility(network) -> dict[int, list[str]]:
    """Determine which processors each layer can run on."""
    compatibility = {}
    for i, layer in enumerate(network.layers):
        procs = ["GPU"]  # GPU can always run layers
        if layer.can_run_on_dla():  # TensorRT: canRunOnDLA()
            procs.append("DLA")
        compatibility[i] = procs
    return compatibility
```

### Step 2: Profile Execution Times

```python
def profile_layer_times(network, processor: str) -> dict[int, float]:
    """
    Profile each layer's execution time on a processor.
    
    Use TensorRT IProfiler API for GPU.
    For DLA, profile groups of layers (black box).
    """
    times = {}
    # ... TensorRT profiling code ...
    return times
```

### Step 3: Profile Transition Costs

```python
def profile_transition_costs(network) -> TransitionModel:
    """
    Empirically measure transition overhead.
    
    Method:
    1. Run all layers on GPU, measure total time
    2. For each potential transition point:
       - Run layers 0..i on GPU, i+1..N on DLA
       - Measure total time
       - Subtract known layer times to isolate transition cost
    """
    # Build regression model: transition_cost = f(tensor_size, layer_type, proc)
    return TransitionModel(...)
```

---

## Solving with Z3

```python
from z3 import Optimize, Int, If, And, Or

def solve_schedule(
    layers: list[Layer],
    processors: list[Processor],
    ect: float,
    max_transitions: int = 3
) -> Schedule:
    """Find optimal schedule using Z3 SMT solver."""
    
    opt = Optimize()
    n_layers = len(layers)
    
    # Decision variables: which processor for each layer
    # 0 = GPU, 1 = DLA
    proc_vars = [Int(f"proc_{i}") for i in range(n_layers)]
    
    # Constrain to valid processor indices
    for v in proc_vars:
        opt.add(And(v >= 0, v < len(processors)))
    
    # Layer compatibility constraints
    for i, layer in enumerate(layers):
        if not layer.can_run_on_dla:
            opt.add(proc_vars[i] == 0)  # Must use GPU
    
    # Transition indicators
    trans_vars = [
        If(proc_vars[i] != proc_vars[i + 1], 1, 0)
        for i in range(n_layers - 1)
    ]
    
    # Limit transitions
    opt.add(sum(trans_vars) <= max_transitions)
    
    # Build time and energy expressions
    total_time_expr = build_time_expression(proc_vars, trans_vars, layers)
    total_energy_expr = build_energy_expression(proc_vars, trans_vars, layers)
    
    # Energy constraint
    opt.add(total_energy_expr <= ect)
    
    # Minimize time
    opt.minimize(total_time_expr)
    
    if opt.check() == sat:
        model = opt.model()
        return {i: processors[model[proc_vars[i]].as_long()] 
                for i in range(n_layers)}
    
    return None  # No feasible schedule
```

---

## Key Insights for Implementation

1. **Transition points matter**: Not all layer boundaries are valid transition points. Check `markOutput` API in TensorRT.

2. **DLA restrictions**: DLA has constraints on batch sizes, certain layer parameters, and can't run all layer types.

3. **Fusion awareness**: TensorRT fuses operations (e.g., Conv+ReLU). Breaking fusions adds overhead.

4. **Pooling after conv**: On DLA, transitioning after pooling layers can actually *reduce* total time because it breaks the pipeline cleanly.

5. **Warm-up profiling**: Account for cold cache misses when profiling layer transitions.

6. **Tensor size correlation**: Transition cost roughly scales with tensor size, but each processor has unique characteristics (DLA has a limited convolution buffer).

---

## References

- Original Paper: [AxoNN (DAC 2022)](https://doi.org/10.1145/3489517.3530572)
- TensorRT Developer Guide: Layer compatibility and DLA
- Z3 SMT Solver: [https://github.com/Z3Prover/z3](https://github.com/Z3Prover/z3)
