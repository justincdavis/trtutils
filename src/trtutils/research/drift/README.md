# DRIFT: Dynamic Intra-Frame Parallelization for Object Detection

Implementation of [DRIFT: Dynamic Intra-Frame Parallelization for Fast Continuous
Object Detection on Multi-Accelerator SoCs](https://doi.org/10.1145/3831252.3834218)
(ICCAD 2026) by Davis, Dagli and Belviranli. It re-implements, and corrects, the
reference implementation [hpsslab/DRIFT](https://github.com/hpsslab/DRIFT), which was
written against older trtutils APIs and took its frame packer from `cv2ext`.

## Summary

Object detection dominates the per-frame cost of a continuous detect-and-track
pipeline, yet it is usually one monolithic pass on the GPU while the SoC's other
accelerators (DLA) sit idle. DRIFT parallelizes detection *within* a frame: the
region that holds the objects that matter is detected on the GPU with an accurate
model, and the rest of the frame is compressed and detected by a cheaper model on
the DLA at the same time.

### Key ideas from the paper

1. **High-priority region** (Section 3.1): the user names the priority classes.
   The padded box around the previous frame's priority detections is the next
   frame's high-priority (HP) region; the rest is the low-priority (LP) region.
   Priority objects found in the LP region grow the HP region on the next frame.
2. **Frame packing** (Section 3.2): the frame is covered by a grid of square
   cells, each counting recent detections. A cell is sampled with probability
   `min(1, max(p_min, exp(-alpha c) + min(1, d / min(FPS, c + 1))))` (Equation 1),
   so empty regions are explored less over time. Sampled cells are grouped into
   dense rectangles and shelf-packed toward a square image (Algorithm 1), with an
   offset table to map boxes back.
3. **Accuracy and latency models** (Section 3.3): per model, a linear estimator
   maps that model's mean detection confidence on a frame to the predicted F1 of
   every model, fitted on a labeled validation set. Latency is characterized end
   to end per model and per concurrent GPU+DLA pair.
4. **Runtime scheduler** (Section 3.4, Algorithm 2): the HP model is the fastest
   GPU model that keeps the HP region's pixel scale and is predicted at least as
   accurate as the baseline; the LP model is the one giving the fastest schedule
   that meets the LP accuracy goal and beats the baseline's latency. Otherwise the
   baseline runs the full frame.

### Results (from paper, MOT17 and KITTI, versus YOLOv10M at 1280)

| Platform | MOT17 latency | MOT17 energy | KITTI latency | KITTI energy |
|---|---|---|---|---|
| Orin AGX | 1.36x faster | 2.8x lower | 1.83x faster | 3.85x lower |
| Orin NX | 1.74x faster | 2.7x lower | 2.3x faster | 3.8x lower |

Recall stays within about 3 percentage points of the baseline.

## How it works (this implementation)

```
frame t ──► hp_region(last detections) ──► DriftScheduler ──► Schedule(hp, lp, parallel)
                 │                                │
                 ▼                                ▼ (lp is None: baseline on the full frame)
          crop HP region ───────────────► HP model, caller thread (GPU) ──┐
          FramePacker.pack(rest) ───────► LP model, worker thread (DLA) ──┤ concurrent
                                          or after HP on the GPU         ▼
                                                     shift / unpack boxes, NMS merge
                                                                  │
                     FramePacker.update(all boxes) ◄──────────────┘──► detections
```

- `Drift` builds every model up front, checks each against its characterization
  (input size, DLA or not), and calls each model a few times so every CUDA graph is
  captured before any concurrent execution. DLA models always run on one worker
  thread, which holds `TRTEngine`'s capture lock while it runs, as SHIFT's
  background builds do.
- The region, the schedule and the split of a frame all come from the previous
  frame's detections, so they always agree.
- Detectors are only called through `end2end`, whose input may change size every
  frame (crops and packed images do) with the `cpu`, `cuda` and `trt` preprocessors,
  with or without CUDA graphs.

## Usage

### Python API

```python
from trtutils.image import Detector
from trtutils.models import YOLOX
from trtutils.research.drift import Drift, DriftConfig, characterize, save_characterization

# the same factories serve characterization and runtime
models = {
    "yolov10m_1280": lambda: Detector("yolov10m_1280.engine"),
    "yolov10n_640": lambda: Detector("yolov10n_640.engine"),
    "yoloxs_480_dla": lambda: YOLOX("yoloxs_480_dla.engine", dla_core=0, cuda_graph=False),
}

# images: BGR arrays or paths; ground_truth: per image, a list of ((x1, y1, x2, y2), class_id)
# both may be generators: images are decoded one at a time and run through every model,
# so a large validation set never sits in memory. The latency loops cycle through a few
# frames held in memory: latency_images, or the first latency_frames (30) labeled images
traits = characterize(models, images, ground_truth, latency_images=video_frames)
save_characterization(traits, "traits.json")

drift = Drift(models, traits, priority_classes={0}, baseline="yolov10m_1280")
for frame in frames:
    detections = drift.end2end(frame)
    print(drift.last.schedule)  # Schedule(hp=..., lp=..., parallel=..., latency_ms=...)
drift.close()

# the reference implementation's hyperparameters
drift = Drift(models, traits, {0}, config=DriftConfig.upstream(["yolov10m_640"]))
```

### Using the pieces on their own

```python
import numpy as np
from trtutils.research.drift import DriftScheduler, FramePacker, LinearEstimator

# frame packing: count detections, pack the likely-relevant cells, map boxes back
packer = FramePacker((1920, 1080), seed=0)
packer.update([(100, 200, 180, 400)])
packed = packer.pack(frame, exclude=(800, 300, 1200, 700))
if packed is not None:
    image, offsets = packed
    boxes, valid = packer.unpack([(10, 10, 60, 90)], offsets)

# accuracy estimation from any aligned per-sample confidences and accuracies
estimator = LinearEstimator(
    {"big": [0.9, 0.4, 0.7], "small": [0.6, 0.2, 0.5]},
    {"big": [0.95, 0.5, 0.8], "small": [0.7, 0.2, 0.55]},
)
estimator.predict("small", 0.45)  # {"big": ..., "small": ...}

# scheduling from characterization data alone, e.g. to replay recorded detections
scheduler = DriftScheduler(traits, baseline="yolov10m_1280")
```

### CLI

```bash
# characterize on labeled images (YOLO-format .txt labels with class ids)
python3 -m trtutils research drift characterize \
    --model yolov10m_1280 yolov10m_1280.engine \
    --model yolov10n_640 yolov10n_640.engine \
    --model yoloxs_480_dla yoloxs_480_dla.engine 0 --input_range yoloxs_480_dla 0 255 \
    --images val/images --labels val/labels --latency_input video.mp4 --output traits.json

# run over a video, people (COCO 0) as the priority class
python3 -m trtutils research drift run \
    --model yolov10m_1280 yolov10m_1280.engine \
    --model yolov10n_640 yolov10n_640.engine \
    --model yoloxs_480_dla yoloxs_480_dla.engine 0 --input_range yoloxs_480_dla 0 255 \
    --characterization traits.json --input video.mp4 --priority 0 --preset paper
```

Every `DriftConfig` field has a `run` flag that overrides the preset.

## Hyperparameters

Defaults are the paper's values. The paper gives no value for the region padding,
the LP accuracy goal or the annealing rate; for those the default is the value of
the reference implementation's documented run, and `DriftConfig.upstream()` uses its
constructor defaults. Every point where the paper and the reference implementation
disagree is its own field.

| Field | Paper (default) | `upstream()` | What it controls |
|---|---|---|---|
| `hp_padding` | 64 | 128 | Pixels around the HP box |
| `hp_square` | False | True | Grow the HP crop's short side toward a square |
| `hp_min_area` | 0.0 | 0.0 | Ignore priority boxes at or below this area |
| `grid_divisor` | 10 | 10 | Cell side = `max(W, H) // grid_divisor` |
| `annealing_rate` | 0.01 | 0.01 | alpha in Equation 1 |
| `annealing_floor` | 0.1 | 0.1 | p_min in Equation 1 |
| `fps` | 30.0 | 30.0 | FPS term in Equation 1 |
| `counter_decrement` | `"all"` | `"all"` | Cell counters: `"all"` decrement after adding hits, `"empty"` decrement only cells without hits [^counter] |
| `dense_groups` | `"rectangles"` | `"blocks"` | Any fully filled rectangle vs only 2x2 / 1x2 / 2x1 blocks |
| `anneal_counter` | `"frames"` | `"packs"` | Equation 1's `c` counts every frame vs only split frames |
| `lp_accuracy_goal` | 0.5 | 0.4 | gamma_lp, compared as `goal / scale` |
| `confidence_metric` | `"mean"` | `"mean"` | Reduction of a frame's confidences |
| `box_size_metric` | `"median"` | `"median"` | Reduction of box sizes in the scale test |
| `accuracy_scale_metric` | `"median"` | `"median"` | Reduction of all F1s that sets the scale (`0.5 / value`) |
| `latency_metric` | `"mean"` | `"mean"` | Reduction of raw latency samples |
| `prediction_window` | 1 | 10 | Frames of F1 predictions averaged |
| `hp_trusted_models` | `()` | caller-supplied | HP models exempt from the accuracy check [^trusted] |
| `lp_exclude_hp_model` | False | True | LP model must differ from the HP model |
| `lp_latency_test` | `"sum"` | `"schedule"` | `lat(hp) + lat(lp)` (Algorithm 2 line 20) vs the schedule's measured latency |
| `merge_iou` | 0.5 | 0.5 | NMS IoU when merging HP and LP detections |

[^counter]: The paper contradicts itself: its formulation decrements only cells without
    detections, but its description ("remain the same with 1 object present") is the
    decrement-all rule, which the reference implementation uses. The default follows
    the description.
[^trusted]: The reference implementation also accepts, without the accuracy check, any
    smaller-input variant of the baseline architecture, found by model name. Name
    conventions are not generic, so the models are listed explicitly.

## Paper erratum: the HP scale test

Algorithm 2 lines 5 and 8 as printed (`s_b = b_m / A_hp`, valid if
`A_m / A_hp <= s_b * b_s`) reduce to `A_m <= b_m * b_s`, where the HP region cancels
out. This implementation always uses the side-length rule of the reference
implementation, which matches the paper's prose: a GPU model `m` may run the HP crop
when it sees the crop at no lower pixel scale than the baseline sees the full frame,
relaxed when objects are large:

```
min(in_w[m] / region_w, in_h[m] / region_h)
    >= min(in_w[base] / W, in_h[base] / H) * (1 - median_box_side / max(W, H))
```

For square inputs this is `imgsz_m / max(region) >= (b_m / max(W, H)) * b_s`.

## Differences from the reference implementation

Bugs fixed (not configurable):

- **No one-frame split lag**: the reference split the frame with the previous
  frame's baseline decision, so a frame scheduled for the baseline could run the
  baseline on the HP crop, and the first split frame after a baseline frame skipped
  the LP model. Here the region, schedule and split are decided together.
- **Model id 0**: the reference treated model id 0 (the smallest GPU model) as "no
  model" in its LP bookkeeping and post-processed baseline frames with the scheduled
  pair's ids. Models are identified by name here, and baseline frames are attributed
  to the baseline for the next prediction.
- **`reset()`** clears the averaged predictions (the reference kept them across videos).
- **Empty packing**: when no cell is sampled, the LP model is skipped, as the paper
  says (the reference ran it on a one-cell black image).
- **F1 of all-wrong predictions** is 0 (the reference scored 1.0 when the counts of
  predictions and labels matched).
- **Class ids**: ground truth is given in the models' class-id space (the reference's
  COCO reader produced 91-class ids against 80-class model outputs).
- **One configuration for profiling and runtime**: characterization uses the runtime's
  own detector factories (the reference profiled YOLOX with a 0-255 input range but
  ran it at 0-1, and benchmarked GPU and DLA pairs with different resize methods).
- **Pair latency**: measured end to end, as the runtime runs it (the reference added
  the DLA model's preprocessing twice and the GPU model's not at all).
- **Contiguous groups**: a group of cells is copied as one region; the reference
  copied each cell's square separately, duplicating the overlap between rows of
  cells and shifting boxes that cross it by up to 15 px at 1080p.

Choices where the paper is silent:

- HP and LP detections merge with class-aware NMS at `merge_iou`.
- Cell counters never go below 0; a box past the last cell's start (a 3 px band at
  1080p) counts toward the last row or column.
- An unmeasured GPU+DLA pair is estimated as the slower of the two models.
- Without an explicit baseline, the GPU model with the highest mean F1 is used.
- Predicted F1 is not clipped to [0, 1].
- Latency is characterized on full frames, so crops and packed images run faster
  than predicted (as in the reference).
- The reference constructor's energy, power and latency budgets only filtered a list
  the live scheduler never read; they are not ported.

## Requirements

- At least one GPU model; DLA models need `dla_core=` and `cuda_graph=False`.
- Do not use the `dla0` / `dla1` preprocessor without `cpu_resize=True`: it builds a
  preprocessing engine per input size, and crops change size every frame.
- All models stay resident, so their engines must fit in memory together.
- YOLOX models expect a 0-255 input range (use `trtutils.models.YOLOX`, or
  `--input_range NAME 0 255` on the CLI).
- A strip up to one cell wide around the HP region is in neither the HP crop nor the
  packed image (cells touched by the region are excluded). This is the paper's design;
  use `hp_padding` to cover it.

## Citation

```bibtex
@inproceedings{davis2026drift,
    title = {DRIFT: Dynamic Intra-Frame Parallelization for Fast Continuous Object Detection on Multi-Accelerator SoCs},
    author = {Justin Davis and Ismet Dagli and Mehmet E. Belviranli},
    booktitle = {IEEE/ACM International Conference on Computer-Aided Design (ICCAD '26)},
    year = {2026},
    doi = {10.1145/3831252.3834218},
}
```
