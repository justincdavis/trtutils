# SHIFT: Context-Aware Multi-Model Object Detection

Implementation of [Context-aware Multi-Model Object Detection for Diversely
Heterogeneous Compute Systems](https://arxiv.org/abs/2402.07415) (DATE 2024) by
Davis and Belviranli. It is ported from, and corrects, the reference implementation
[shiftdetector](https://github.com/justincdavis/shiftdetector).

## Summary

Edge SoCs such as NVIDIA Jetson have several accelerators (GPU, DLA), and a
detector compiled for each is a different trade-off of accuracy, energy and
latency. No single (model, accelerator) pair is best for every frame: an easy
scene can be handled by a small model on the DLA, while a hard one needs a large
model on the GPU. SHIFT picks the pair per frame from the scene context.

### Key ideas from the paper

1. **Offline characterization**: every pair is run on labeled images, recording
   its confidence and accuracy (IoU) per image, its latency and energy per
   inference, and its loading cost.
2. **Confidence graph**: one model's confidence predicts how accurate every other
   model would be. Nodes are (model, confidence bin) pairs holding the mean
   accuracy of that bin; edges link bins that occur on the same images. A
   shortest-path search within a distance threshold turns one node into an
   accuracy prediction for every model, replacing a learned accuracy predictor
   with a lookup.
3. **Context-aware scheduling**: the current pair is kept while
   `similarity * confidence >= accuracy_threshold`, where similarity combines
   the normalized cross-correlation (NCC) between consecutive frames and between
   consecutive detection boxes. Otherwise the predicted accuracies are averaged
   over the last `momentum` reschedules, pairs below the goal accuracy are
   dropped, and the best weighted score of accuracy, energy and latency wins.
4. **Dynamic model loading**: pairs are loaded on demand under a memory budget,
   in the background, and the least recently used pairs are evicted.

### Results (from paper, Xavier NX + OAK-D, versus YOLOv7 on the GPU)

| Metric | Change |
|--------|:------:|
| Latency | 2.8x lower |
| Energy | 7.5x lower |
| Successful frames (IoU >= 0.5) | 0.97x |
| Average IoU | 0.97x |

## How it works (this implementation)

```
labeled images ──► characterize() ──► {pair: ModelTraits} ──► JSON
                    (per pair: load, run, benchmark)
                                                │
                                                ▼
frame ──► Shift.end2end() ──► current pair ──► detections ──► caller
              ▲                                   │
              │                                   ▼
              │                         ShiftScheduler
              │                   gate: similarity × confidence
              │                   ConfidenceGraph.predict()
              │                   momentum, goal filter, score
              │                                   │
              └──── switch when loaded ◄──── target pair
                    (background load, LRU eviction under memory_budget)
```

## Usage

### Python API

```python
from trtutils.image import Detector
from trtutils.research.shift import Shift, characterize, save_characterization

models = {
    "yolov10n_gpu": lambda: Detector("yolov10n.engine"),
    "yolov10n_dla": lambda: Detector("yolov10n_dla.engine", dla_core=0, cuda_graph=False),
    "yolov11m_gpu": lambda: Detector("yolov11m.engine"),
}

# images: BGR arrays or paths (any iterable); ground_truth: per image, a list of
# (x1, y1, x2, y2) boxes. Paths are decoded in the background as each pair runs.
traits = characterize(models, images, ground_truth)
save_characterization(traits, "traits.json")

shift = Shift(models, traits)  # paper hyperparameters
# shift = Shift(models, traits, memory_budget=1500)  # load pairs on demand within 1.5 GB
for frame in frames:
    detections = shift.end2end(frame)  # list of ((x1, y1, x2, y2), score, class_id)
    print(shift.current)  # pair for the next frame
```

Pairs are built by the factories you pass, so anything a `Detector` accepts
(preprocessor, thresholds, schemas) is per pair. Engines with DLA layers need
`dla_core` and `cuda_graph=False`.

`characterize` builds, runs, benchmarks and frees one pair at a time, and every
pair runs over every labeled image. Pass paths for large datasets: during each
pair's image pass, `decode_workers` threads (default 2) decode the next
`prefetch` images (default 4) while the pair runs on the current one, and each
frame is dropped after use. Memory holds at most `prefetch + 2` frames instead of
the whole set; each image is decoded once per pair (about 3 ms for a 640x480 JPEG
and 16 ms for a 1080p one on an Orin CPU), overlapped with inference. The threads
are joined before the pair is benchmarked, so decoding never skews latency or
energy. Arrays are used as given and stay in memory. The CLI passes paths.

### Using the confidence graph on its own

`ConfidenceGraph` needs only per-sample confidences and accuracies, aligned by
sample index. The models do not have to be detectors, and "accuracy" can be any
per-sample quality metric.

```python
from trtutils.research.shift import ConfidenceGraph

graph = ConfidenceGraph(
    confidences={"big": [0.9, 0.8, 0.2], "small": [0.7, 0.4, 0.0]},
    accuracies={"big": [0.85, 0.80, 0.30], "small": [0.70, 0.35, 0.0]},
    num_bins=10,
    distance_threshold=0.5,
)
graph.predict("small", 0.65)  # {'big': 0.85, 'small': 0.7}
graph.nodes  # {(model, bin): expected accuracy}
graph.edges  # {(model, bin): {(model, bin): traversal cost}}
```

### CLI

```bash
# characterize on labeled images (YOLO-format .txt labels, one per image stem)
trtutils research shift characterize \
    --model yolov10n_gpu yolov10n.engine \
    --model yolov10n_dla yolov10n_dla.engine 0 \
    --model yolov11m_gpu yolov11m.engine \
    --images val/images --labels val/labels \
    --output traits.json

# run over a video
trtutils research shift run \
    --model yolov10n_gpu yolov10n.engine \
    --model yolov10n_dla yolov10n_dla.engine 0 \
    --model yolov11m_gpu yolov11m.engine \
    --characterization traits.json \
    --input video.mp4 \
    --preset paper --memory_budget 1500
```

`--model NAME ENGINE [DLA_CORE]` is repeated once per pair. Every `ShiftConfig`
field can be overridden by a flag of the same name (e.g. `--momentum 10`).

## Hyperparameters

`ShiftConfig()` holds the paper's values (Table III). Where the paper and
shiftdetector disagree, the difference is a field, and
`ShiftConfig.shiftdetector()` (CLI `--preset shiftdetector`) gives the reference
implementation's values.

| Field | Paper (default) | shiftdetector | Description |
|-------|:---------------:|:-------------:|-------------|
| `accuracy_threshold` | 0.25 | 0.5 | Goal accuracy; gate threshold and candidate filter |
| `momentum` | 30 | 10 | Predictions averaged per pair |
| `distance_threshold` | 0.5 | 1.0 | Maximum graph path cost to a neighbor |
| `accuracy_weight` | 1.0 | 0.85 | Score weight of predicted accuracy |
| `energy_weight` | 0.5 | 0.10 | Score weight of inverted, normalized energy |
| `latency_weight` | 0.5 | 0.10 | Score weight of inverted, normalized latency |
| `similarity` | `min` | `product` | How frame NCC and box NCC combine |
| `outlier_percentile` | 100 | 80 | Percentile of a node's edge counts used to normalize them |
| `min_connects` | 0 | 1 | Drop edges seen this many times or fewer |
| `num_bins` | 10 | 10 | Confidence bins per model |
| `confidence_metric` | `mean`* | `max` | Reduction of a frame's detection confidences |
| `accuracy_metric` | `mean`* | `max` | Reduction of a frame's per-ground-truth IoUs |

\* The paper and shiftdetector are single-object (one box per frame), which
`max` reproduces: the top detection's confidence and its IoU. The default `mean`
is this implementation's multi-object generalization. It generalizes to many
objects per frame: per image it records every detection's confidence and, for
every ground-truth box, the best IoU of any detection (class agnostic), and the
metrics choose how a frame reduces to one value. An image without ground truth
scores 1.0 if the pair also detects nothing and 0.0 otherwise.

The characterization stores raw per-image values, so every field above can be
changed without characterizing again.

## Differences from shiftdetector

These shiftdetector behaviors look unintended and are fixed rather than
configurable:

- NCC is standard zero-mean NCC; shiftdetector computed `x - mean / std` due to
  operator precedence.
- Edge counts above the outlier percentile are clamped; shiftdetector computed
  the clamp, discarded it, and so dropped those edges.
- A neighbor's distance is its shortest-path cost; shiftdetector used the cost
  of the single BFS-tree edge into it.
- Only pairs the graph predicts for the current frame are candidates;
  shiftdetector also scored stale predictions left from earlier frames.
- Box NCC compares the previous frame's box crop with the current one;
  shiftdetector cropped both frames at the current box.
- Edges are directed, since normalization is per source node; shiftdetector
  stored them in an undirected graph, where the last direction written won.

Other implementation choices where the paper is silent:

- Confidence bins are `ceil(confidence * num_bins)`, so bin 0 is exactly "no
  detection".
- Prediction weights are `max(((threshold - distance) / threshold) ** 2, 1e-8)`,
  as in shiftdetector.
- Energy and latency are normalized by the maximum and inverted, `1 - value / max`,
  so the most expensive pair scores 0. Energy is 0.0 for every pair when not measured (non-Jetson),
  so only accuracy and latency matter there.
- On a frame whose (pair, bin) was never seen during characterization, the
  current pair is kept.
- `Shift` starts on the pair with the highest mean characterized accuracy.
- Latency and energy are engine-only (`benchmark_engine` /
  `jetson.benchmark_engine`, energy from tegrastats `VDD_TOTAL`); loading memory
  is the `cudaMemGetInfo` delta around building the pair, which on Jetson's
  unified memory also includes host allocations.

## Requirements

- **Hardware**: any CUDA GPU; energy measurement and DLA pairs need an NVIDIA
  Jetson.
- **Dependencies**: none beyond trtutils (numpy and OpenCV).
- **Background loading**: TensorRT and NVRTC hold the GIL for parts of building a
  detector, so a background load still stalls the serving pair for a few frames.
  On an Orin AGX, building a YOLOv10n detector blocks for ~490 ms synchronously;
  in the background the current pair keeps serving (~60 frames), with the
  longest single-frame stall ~210 ms. A background build holds trtutils' CUDA graph capture
  lock, because building an engine uses the legacy default stream, which would break
  a capture running in the serving thread; a frame that needs a capture waits for
  the build to finish.

## Citation

```bibtex
@misc{davis2024shift,
    title = {Context-aware Multi-Model Object Detection for Diversely Heterogeneous Compute Systems},
    author = {Justin Davis and Mehmet E. Belviranli},
    year = {2024},
    eprint = {2402.07415},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url = {https://arxiv.org/abs/2402.07415},
}
```
