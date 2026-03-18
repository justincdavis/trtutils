# AxoNN: Energy-Aware Multi-Accelerator Neural Network Inference

Implementation of [AxoNN: Energy-Aware Execution of Neural Network Inference on
Multi-Accelerator Heterogeneous SoCs](https://doi.org/10.1145/3489517.3530572)
(DAC 2022) by Dagli et al.

## Summary

AxoNN solves a practical problem on NVIDIA Jetson SoCs: **how to split neural
network layers between the GPU and DLA to minimize latency under an energy
budget**.

The GPU is fast but power-hungry. The DLA is slow but energy-efficient. AxoNN
profiles every layer on both accelerators, models the cost of transitioning
between them (cache flushes, cold misses), and uses constraint optimization
(Z3 SMT solver) to find the best layer-to-accelerator mapping.

### Key ideas from the paper

1. **Per-layer profiling** — Each layer is characterized for execution time and
   energy on both GPU and DLA independently.
2. **Transition cost modeling** — Switching between GPU and DLA incurs overhead
   proportional to the output tensor size (cache writeback + cold start).
   DLA's private convolution buffer makes its transition costs non-linear.
3. **Constrained optimization** — Given an Energy Consumption Target (ECT), the
   solver minimizes total execution time subject to `total_energy <= ECT`. A
   binary search over the time budget drives the solver toward the optimal
   Pareto point.
4. **Pipeline-aware scheduling** — Breaking DLA hardware pipelines
   (conv → activation → pooling) at the wrong layer adds overhead; the model
   accounts for this.

### Results (from paper, Xavier AGX)

| Model     | Time prediction accuracy | Energy prediction accuracy |
|-----------|:------------------------:|:--------------------------:|
| VGG-19    | up to 98%                | up to 97%                  |
| VGG-16    | up to 98%                | up to 97%                  |
| AlexNet   | up to 97%                | up to 98%                  |
| ResNet-50 | up to 95%                | up to 96%                  |
| ResNet-18 | up to 96%                | up to 97%                  |
| GoogleNet | up to 96%                | up to 97%                  |

## How it works (this implementation)

```
ONNX model
    │
    ▼
┌──────────────────┐
│  inspect layers   │  ← extract layer graph, DLA compatibility, tensor sizes
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│ GPU    │ │ DLA    │   ← build temp engines, profile with tegrastats
│ profile│ │ profile│     (latency per layer + total board energy)
└───┬────┘ └───┬────┘
    └────┬─────┘
         ▼
┌──────────────────┐
│  Z3 solver       │  ← binary search: minimize time s.t. energy <= ECT
│  (constraint     │     with transition cost model
│   optimization)  │
└────────┬─────────┘
         ▼
┌──────────────────┐
│  build final     │  ← TensorRT engine with per-layer GPU/DLA assignments
│  engine          │
└──────────────────┘
```

## Usage

### Python API

```python
from trtutils.builder import ImageBatcher
from trtutils.research.axonn import build_engine

batcher = ImageBatcher(
    image_dir="calibration_images/",
    shape=(640, 640, 3),
    dtype=np.float32,
)

time_ms, energy_mj, transitions, gpu_layers, dla_layers = build_engine(
    onnx="model.onnx",
    output="model.engine",
    calibration_batcher=batcher,
    energy_ratio=0.8,       # ECT = 80% of GPU-only energy
    max_transitions=1,      # max GPU<->DLA switches
    verbose=True,
)

print(f"Time: {time_ms:.2f}ms, Energy: {energy_mj:.2f}mJ")
print(f"GPU layers: {gpu_layers}, DLA layers: {dla_layers}")
```

### CLI

```bash
# with calibration images
trtutils research axonn build \
    --onnx model.onnx \
    --output model.engine \
    --calibration_dir calibration_images/ \
    --input_shape 640 640 3 \
    --input_dtype float32 \
    --energy_ratio 0.8 \
    --max_transitions 1 \
    --verbose

# with synthetic calibration data (no images needed)
trtutils research axonn build \
    --onnx model.onnx \
    --output model.engine \
    --energy_ratio 0.7 \
    --max_transitions 2 \
    --verbose
```

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `energy_ratio` | 0.8 | ECT as fraction of GPU-only energy. Lower = more DLA layers. |
| `energy_target` | None | Explicit ECT in mJ (overrides `energy_ratio`). |
| `max_transitions` | 1 | Max GPU↔DLA switches. Paper shows 1-2 is usually optimal. |
| `profile_iterations` | 1000 | Profiling iterations per engine for stable measurements. |

## Requirements

- **Hardware**: NVIDIA Jetson (Xavier AGX, Orin) with GPU + DLA
- **Dependencies**: `z3-solver` (installed via `pip install trtutils[axonn]`)
- **TensorRT**: with DLA support (Jetson JetPack)

## Citation

```bibtex
@inproceedings{dagli2022axonn,
    author = {Dagli, Ismet and Cieslewicz, Alexander and McClurg, Jedidiah and Belviranli, Mehmet E.},
    title = {AxoNN: energy-aware execution of neural network inference on multi-accelerator heterogeneous SoCs},
    year = {2022},
    isbn = {9781450391429},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3489517.3530572},
    doi = {10.1145/3489517.3530572},
    booktitle = {Proceedings of the 59th ACM/IEEE Design Automation Conference},
    pages = {1069–1074},
    location = {San Francisco, California},
    series = {DAC '22}
}
```
