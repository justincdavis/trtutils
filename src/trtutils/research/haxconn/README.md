# HaX-CoNN: Contention-Aware Concurrent DNN Execution

Implementation of [HaX-CoNN: Shared Memory-Contention-Aware Concurrent DNN Execution for
Diversely Heterogeneous System-on-Chips](https://arxiv.org/abs/2308.05869) (PPoPP 2024) by
Dagli and Belviranli. The reference repository
[ismetdagli/HaX-CoNN](https://github.com/ismetdagli/HaX-CoNN) has no license and holds one
solver hardcoded to GoogleNet + AlexNet with hand-typed inputs, so this is a fresh
implementation from the paper and from the PCCS paper it relies on.

## Summary

Autonomous systems run several DNNs at once (detection, segmentation, tracking). A Jetson
has two kinds of DNN accelerators, the GPU and the DLA, which share one memory system. Running
each DNN whole on one accelerator leaves performance on the table, and naive layer-level
splitting ignores two costs: moving between accelerators, and the slowdown the accelerators
cause each other through shared memory. HaX-CoNN maps every layer group of every DNN to an
accelerator with both costs in the model and a solver that proves its schedule optimal.

### Key ideas from the paper

1. **Layer grouping** (Section 3.1): layers that TensorRT fuses, or whose transition would
   force reformatting, stay together; the boundaries left are the transition points.
2. **Layer and transition characterization** (Section 3.2): every group is timed on each
   accelerator, and the cost of moving after it (GPU to DLA and DLA to GPU) is measured with
   TensorRT's `markOutput` / `addInput`.
3. **Contention model** (Section 3.3): each group's standalone memory throughput feeds
   PCCS (Xu et al., MICRO 2021), a piecewise model of the speed a processor keeps given its
   own demand and the external demand of the others.
4. **Formulation** (Section 3.4): device variables, start and end times, transition costs,
   mutual exclusion per accelerator (Eq. 9), and contention over the intervals in which
   groups overlap (Eqs. 7-8). Objectives: maximum summed throughput (Eq. 10) or minimum
   maximum latency (Eq. 11). Solved with Z3.
5. **D-HaX-CoNN** (Sections 3.5 and 5.3): the set of DNNs that run together changes as
   the system moves between phases. Each new set starts on the best naive schedule while Z3
   runs on one CPU core, and the runtime switches to the best schedule found so far 25 ms,
   100 ms, 250 ms, 500 ms and 1.5 s after the solver starts, until no better one exists. A
   schedule found for a set is reused while the set runs.

### Results (from paper)

Up to 32% lower latency and 29% higher throughput than the best baseline (GPU only, naive
GPU & DLA, Herald, H2H) on Xavier AGX, Orin AGX and Snapdragon 865; it never does worse
than a naive schedule because the solver can pick one. D-HaX-CoNN reaches the optimum in
1.3-5.8 s with Z3 on one CPU core, at under 2% overhead on the running DNNs.

### Results (this implementation, Orin AGX 64GB, TensorRT 8.6, INT8)

Two DNNs started together each round, 300 rounds, median of the slower DNN in ms (12
layer groups each, `contention="overlap"`, clocks not locked):

| Pair | GPU only | Naive GPU & DLA | HaX-CoNN (predicted) |
| --- | --- | --- | --- |
| ResNet34 + VGG16 | 1.96 | 1.70 | 1.70 (1.56) |
| ResNet18 + VGG16 | 1.78 | 1.68 | 1.68 (1.54) |
| ResNet18 + ResNet34 | 0.89 | 0.81 | 0.81 (0.59) |

For these CNNs, which run well on the INT8 DLA, the optimal schedule is the naive split;
the solver proves it in under a second, and D-HaX-CoNN gets there in under 0.5 s. In an
earlier run with differently built engines, the schedule for ResNet18 + ResNet34 split
ResNet34 across both accelerators and beat both baselines by over 20%, but run-to-run
variance without locked clocks is of that order, so lock them (`sudo jetson_clocks`) for
comparisons. Predictions of the slowest DNN are within 10% when it runs on the GPU; DLA
predictions run high.

D-HaX-CoNN on the paper's Section 5.3 scenario (three sets of DNNs for 10 s each, twice;
all 30 engines prebuilt in 6.5 min; Z3 pinned to one core; slowest DNN's median round, ms):

| Set | Proven optimal after | While solving | After | Overhead |
| --- | --- | --- | --- | --- |
| ResNet34 + VGG16 | 0.34 s | 1.70 | 1.69 | +0.6% |
| ResNet18 + ResNet34 + VGG16 | not within 10 s | 1.88-1.97 | 1.84-1.88 | +3% |
| ResNet18 + ResNet34 | 0.30 s | 0.81-0.85 | 0.82 | under 4% |

Every set's optimum is again its naive schedule, so no switch was needed; the second time a
set ran it started on its proven schedule without a search. Changing the set paused the
DNNs for 10-15 ms (45-51 ms the first time a set needs an engine that is not loaded yet).
For the three-DNN set Z3 needs 45 s to prove optimality, while `solver="exhaustive"`
scores all 990 schedules within one transition per DNN in 0.44 s beside the running DNNs
(at +15% round time while it runs): for small transition-limited problems, the exhaustive
solver is the faster D-HaX-CoNN search. The paper reports 1.3-5.8 s to the optimum and
under 2% overhead.

## How it works (this implementation)

```
offline, per DNN            offline, per board          online
----------------            ------------------          ------
characterize(onnx)          calibrate()                 HaxConnScheduler(traits, models)
  inspect_onnx_layers         conv probes, GPU + DLA      Z3: paper constraints, no contention
  group_layers                standalone EMC % each        -> proposes an assignment
  slice_hook per group        GPU x DLA co-runs           simulate(): exact timeline with
   -> GPU / DLA engines        -> relative speeds            transitions, queueing, PCCS
  profile: time, in/out     PCCSModel.from_calibration    exclude it, repeat until unsat
  tegrastats: EMC %                                     HaxConn: one mixed engine per DNN,
   -> ModelTraits                                         all DNNs started together
```

Z3 keeps the paper's formulation without the contention terms, which makes it a lower bound
on every schedule. It is asked for an assignment whose bound beats the best schedule so far;
that assignment is scored exactly by `simulate`, excluded, and the loop continues until Z3
answers unsat, which proves the best schedule optimal, or the time budget ends. Overlap
contention inside Z3 itself (Eq. 7) times out even for two small DNNs (Z3 4.14 answers
`unknown` after 15-20 s per check for 10 + 8 groups), which is also why the reference solver
dropped it.

**D-HaX-CoNN.** `HaxConn(dynamic=True)` holds a pool of characterized DNNs;
`set_active(names)` picks the ones that run together and may be called between any two
rounds. A new set starts at once on the best schedule found for it before, or else on the
best naive schedule, while two background threads run: one iterates `solve_iter()` (Z3 on
one core with `solver_core`), the other loads the best schedule's engines at each
`update_ms` time and hands them over between two rounds. Changing the set interrupts the
old search. An engine depends only on its own DNN's assignment, so `prebuild_engines`
builds every assignment within the transition limit once (9-11 per DNN for the CNNs below)
and `build="prebuilt"` makes every switch a load; loaded engines stay in memory
(`max_engines`), so a set that comes back switches without loading anything.

`simulate` releases all DNNs together. A run of groups of one DNN on one accelerator holds
the accelerator (a DLA subgraph cannot be preempted); DNNs waiting for a busy accelerator
queue in arrival order. Moving between accelerators costs the transition-out of the group
before and the transition-in of the group after. While both accelerators are busy, each
group progresses at `1 / slowdown(own demand, other group's demand)`.

## Usage

### Python API

```python
from trtutils.research.haxconn import (
    HaxConn,
    HaxConnScheduler,
    PCCSModel,
    calibrate,
    characterize,
    save_traits,
)

models = {"resnet34": "resnet34.onnx", "vgg16": "vgg16.onnx"}

# offline: per DNN, and once per board (memory throughput needs tegrastats as root)
traits = {
    name: characterize(onnx, max_groups=12, int8=True, memory=True) for name, onnx in models.items()
}
contention = {acc: PCCSModel.from_calibration(c) for acc, c in calibrate(int8=True).items()}

# solve and compare with the naive schedules
scheduler = HaxConnScheduler(traits, contention)
schedule = scheduler.solve()
print(schedule.assignments, schedule.latency_ms, scheduler.baselines())

# run; pass data_batchers for usable INT8 outputs
runtime = HaxConn(models, traits, contention, schedule=schedule, engine_dir="engines")
outputs = runtime.mock_execute()  # or runtime.execute({name: inputs, ...})
print(runtime.last)  # per-DNN latency of the round, ms
runtime.close()

# D-HaX-CoNN: build every engine a schedule can use, then change the running set at will
pool = {**models, "resnet18": "resnet18.onnx"}
pool_traits = {
    **traits,
    "resnet18": characterize("resnet18.onnx", max_groups=12, int8=True, memory=True),
}
prebuild_engines(pool, pool_traits, "engines")
runtime = HaxConn(
    pool,
    pool_traits,
    contention,
    engine_dir="engines",
    active=["resnet34", "vgg16"],
    dynamic=True,
    build="prebuilt",
    solver_core=11,
)
runtime.mock_execute()  # starts naive, switches as Z3 finds better schedules
runtime.set_active(["resnet18", "resnet34"])  # a new phase: a new search
print(runtime.schedule, runtime.searching, runtime.schedules)
runtime.close()
```

### Using the pieces on their own

```python
from trtutils.inspect import inspect_onnx_layers
from trtutils.research.haxconn import group_layers, slice_hook, simulate, fit_pccs

groups = group_layers(inspect_onnx_layers("model.onnx"), max_groups=8)
build_engine("model.onnx", "group3.engine", hooks=[slice_hook(groups[3].first, groups[3].last)])
model = fit_pccs(demand, external, relative_speed)  # any calibration matrix
latency = simulate(list(traits.values()), assignments, contention)
```

### CLI

```bash
python3 -m trtutils research haxconn characterize --onnx resnet34.onnx \
    --output resnet34.json --int8 --memory --max_groups 12
python3 -m trtutils research haxconn calibrate --output calibration.json --int8
python3 -m trtutils research haxconn solve --model resnet34 resnet34.json \
    --model vgg16 vgg16.json --calibration calibration.json --output schedule.json
python3 -m trtutils research haxconn run --model resnet34 resnet34.onnx resnet34.json \
    --model vgg16 vgg16.onnx vgg16.json --calibration calibration.json \
    --schedule schedule.json --baselines
# D-HaX-CoNN: every engine a schedule can use, then three sets for 10 s each (paper Section 5.3)
python3 -m trtutils research haxconn prebuild --model resnet18 resnet18.onnx resnet18.json \
    --model resnet34 resnet34.onnx resnet34.json --model vgg16 vgg16.onnx vgg16.json \
    --engine_dir engines
python3 -m trtutils research haxconn dynamic --model resnet18 resnet18.onnx resnet18.json \
    --model resnet34 resnet34.onnx resnet34.json --model vgg16 vgg16.onnx vgg16.json \
    --calibration calibration.json --engine_dir engines --build prebuilt --solver_core 11 \
    --workload resnet34,vgg16 --workload resnet18,resnet34,vgg16 --workload resnet18,resnet34 \
    --period 10 --cycles 2 --output timeline.json
```

## Hyperparameters

| Field | Paper (default) | `upstream()` | What |
| --- | --- | --- | --- |
| `objective` | `"latency"` | `"latency"` | Eq. 11 min max latency, or `"throughput"` (Eq. 10) |
| `contention` | `"overlap"` | `"static"` | Eq. 7 overlap-aware slowdown; a fixed per-group slowdown at saturated external demand [1]; or `"none"` |
| `max_transitions` | `1` [2] | `1` | Transitions per DNN; `None` is unbounded |
| `solver` | `"z3"` | `"z3"` | Or `"exhaustive"` within `max_transitions` |
| `time_budget` | `30.0` s | `30.0` s | Search budget; `None` searches until proven optimal |
| `latency_metric` | `"mean"` | `"mean"` | Reduction of raw time samples |
| `memory_metric` | `"median"` | `"max"` [3] | Reduction of raw EMC samples |
| `update_ms` | `(25, 100, 250, 500, 1500)` | same [4] | D-HaX-CoNN switch times after a search starts; every improvement after the last; `None` switches on every improvement |

[1] The reference solver applies a group's slowdown whether or not anything runs beside it.
[2] The paper states no bound, but every schedule it reports has at most one transition per
DNN and the reference solver enforces one.
[3] The reference EMC script keeps each run's peak utilisation.
[4] The reference implementation has no dynamic mode.

Runtime options of `HaxConn` that the paper does not fix: `build` ("background" builds a
better schedule's engines while the DNNs run, "prebuilt" switches only to engines already
built), `solver_core`, `max_engines`.

## Paper erratum

- **Eq. 8** (overlap length) as printed has a self-contradictory first case and a second case
  that repeats one clause, and returns the full duration when nothing overlaps. The intended
  quantity is the overlap of two intervals, `max(0, min(e_i, e_j) - max(s_i, s_j))`, which
  `simulate` computes exactly by advancing time between events.
- **Eq. 7** divides by the number of overlapping layers without explanation; read as a
  time-weighted average slowdown, it is what `simulate` integrates.
- **PCCS Eq. 5** writes its first condition as `TBWDC <= y`; by analogy with Eq. 3 it is
  `TBWDC <= x + y`.
- **Section 5.3** lists the schedule updates as 25 ms to 1.5 s after Z3 starts, but reports
  convergence after up to 5.8 s, so updates must continue after 1.5 s; here every
  improvement after the last `update_ms` time is switched in. It also attributes the
  slowest convergence to "the latter pair" with three DNNs, while the three-DNN experiment
  is the second of the three in its order (Table 6 experiments 2, 5, 1).

## Differences from the reference implementation

Bugs and simplifications of the reference solver (`src/z3_solver_multi_dnn.py`) that this
implementation does not copy:

- It models exactly GoogleNet + AlexNet with hand-typed inputs, two of them marked "dummy";
  any number of DNNs is supported here, from measured traits.
- Its transition cost reads the loop's global `i`, ignores direction (only the GPU array is
  used), and takes the predecessor's duration from the wrong accelerator at the transition.
  Here each direction has its own in/out costs.
- Its slowdown multiplies only predecessor durations and ignores the co-runner; mutual
  exclusion and the final latency use unslowed times. `simulate` slows every group by what
  actually runs beside it (or, with `contention="static"`, every group always).
- Its PCCS branch for normal-region GPU kernels (`100 / (100 - x)`) matches no PCCS equation
  and is discontinuous; the full PCCS model is implemented here.
- It has no throughput objective, time budget or dynamic mode.

Choices where the papers are silent:

- **How D-HaX-CoNN switches schedules**: the paper does not say whether engines are prebuilt
  or how a new schedule takes effect. Here every DNN runs as one monolithic engine per
  assignment (as in static mode), switched between two rounds; engines come from
  `prebuild_engines` or are built in the background. A new set restarts the search from
  the best schedule known for it, and a set with a proven-optimal schedule runs it without a
  search.
- **Solver beside the DNNs**: scoring schedules is pure Python, and a search thread that
  scores back to back holds the interpreter lock, so the DNN worker threads wait up to 5 ms
  per handover: rounds of the three-DNN set took 34-67 ms instead of 1.9 ms while the
  exhaustive solver ran. The runtime's searches sleep 0.1 ms after each scored schedule
  (`HaxConnScheduler(pause=...)`); `sleep(0)` is worse, since the solver takes the lock
  straight back. Z3's own checks run in C without the lock.

- **Transition points** need exactly one data tensor crossing the cut and no fused engine
  layer spanning it; constant-derived tensors (shapes, reshape targets) do not count.
- **Group timing**: GPU times come from the full GPU engine's per-layer profile (as in the
  paper), because slices built on their own lose TensorRT's optimizations across their
  boundaries (VGG16's slices summed to 25% more than the whole engine on Orin). A full
  DLA engine reports one opaque subgraph, so DLA times come from engines sliced out of the
  network (`slice_hook`). Each slice's profile splits into leading reformats
  (transition-in), compute and trailing reformats (transition-out); profiled times are
  never scaled above the unprofiled wall time.
- **One DNN per accelerator at a time** (Eq. 9): two DNNs on the GPU really share it,
  so in schedules that put both on the GPU the simulator finishes one early and the other
  late, while the slowest finishes at about the right time.
- **DLA launch cost**: each DLA task pays about 0.05-0.12 ms on Orin. It is measured from an
  engine spanning several DLA groups and moved into each group's DLA transition-in, so a run
  of DLA groups pays it once.
- **Memory throughput** is tegrastats' EMC utilisation (percent, about every 30 ms) while the
  group engine runs back to back, divided by the accelerator's duty cycle for groups shorter
  than their launch. Lock the clocks (`sudo jetson_clocks`) so percentages are comparable.
- **PCCS fit**: the minor and intensive regions follow the PCCS paper's steps 1-2; TBWDC,
  CBP and the rate are fit by least squares to Eqs. 3-5 instead of its turn-point
  heuristics, which need a fine, noise-free grid (on this Orin they missed known models by
  up to 14 points; least squares by 2). A DLA has no minor region (as in the PCCS paper).
- **Calibration probes** are stacks of random-weight convolutions whose depth grows as their
  input shrinks, so every probe does about the same work: tiny probes measured CPU and
  launch interference between the two loops instead of memory contention.
- **INT8 characterization** uses nominal dynamic ranges (`nominal_ranges_hook`): INT8 kernels
  run as fast with any scale. Runtime engines are calibrated with the given data batchers.
- **Throughput objective in Z3**: `1000 / T` is bounded above by the maximum of its chords,
  which keeps the model linear and the bound valid.
- **Eq. 9's epsilon** is not implemented: the paper gives no value and the reference has none.

## Requirements

- `pip install "trtutils[haxconn]"` (Z3).
- A Jetson with a DLA for the DLA side; static input shapes.
- Root or passwordless sudo for tegrastats when measuring memory throughput or calibrating.

## Citation

```bibtex
@inproceedings{dagli2024haxconn,
  title     = {HaX-CoNN: Shared Memory-Contention-Aware Concurrent DNN Execution for
               Diversely Heterogeneous System-on-Chips},
  author    = {Dagli, Ismet and Belviranli, Mehmet E.},
  booktitle = {Proceedings of the 29th ACM SIGPLAN Annual Symposium on Principles and
               Practice of Parallel Programming (PPoPP)},
  year      = {2024}
}
```
