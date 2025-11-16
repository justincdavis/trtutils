Here’s a **developer-oriented methodology summary and walkthrough** of the paper *Remix: Flexible High-resolution Object Detection on Edge Devices with Tunable Latency* (ACM MobiCom ’21).
This includes a **dataflow diagram-style breakdown** and **pseudo-code guidance** for each core module, so an engineer could implement it end-to-end.

---

## 🎯 Objective

Remix enables **high-resolution object detection on edge devices** while meeting a **user-specified latency budget**.
It adapts computation per image region by:

* **Non-uniform image partitioning**
* **Selective neural network execution**
* **Dynamic runtime control to balance accuracy and latency**

---

## 🧩 System Dataflow Overview

```
          ┌────────────────────────────┐
          │     Historical Frames      │
          └─────────────┬──────────────┘
                        │
        ┌───────────────▼────────────────┐
        │  Adaptive Partition Module      │
        │  (Offline / Initialization)     │
        ├────────────────────────────────┤
        │  1. NN Profiling                │
        │  2. Object Distribution Extract │
        │  3. Performance Estimation      │
        │  4. Partition Plan Generation   │
        └───────────────┬────────────────┘
                        │
               Partition Plan Pool
                        │
        ┌───────────────▼────────────────┐
        │     Selective Execution         │
        │     (Online / Runtime)          │
        ├────────────────────────────────┤
        │  1. Partition Selection (AIMD)  │
        │  2. Batch Execution             │
        │  3. Plan Control (PID loop)     │
        │  4. BBox Merging (NMS)          │
        └───────────────┬────────────────┘
                        │
                  Final Detections
```

---

## 1️⃣ Adaptive Partition Module (Offline)

**Goal:** Generate a set of optimal image partition plans balancing accuracy vs latency.

### Step 1: Neural Network Profiling

Profiles latency and size-specific accuracy for each NN.

**Data:**

* Candidate networks (e.g., EfficientDet-D0...D6)
* Device runtime performance (latency, batch size)
* Dataset for accuracy estimation (e.g., MS-COCO)

**Outputs:**

* `APn`: accuracy vector of NN `n` across object size bins
* `Lb_n`: latency per batch

**Pseudo-code:**

```python
def profile_networks(networks, device):
    profiles = {}
    for n in networks:
        latency = measure_latency(n, device)
        acc_vector = evaluate_accuracy_by_size(n, dataset='MSCOCO')
        profiles[n] = {'latency': latency, 'acc_vector': acc_vector}
    return profiles
```

---

### Step 2: Object Distribution Extraction

Estimates spatial and size distribution of objects in the view using *historical frames*.

**Technique:**

* Run a high-accuracy “oracle” model (e.g., UP-D7) on historical frames.
* Collect per-block distributions of detected object sizes.

**Output:**

* Distribution vector `F_p` for each region `p` of the image.

**Pseudo-code:**

```python
def extract_object_distribution(frames, oracle_model):
    all_detections = [oracle_model.detect(f) for f in frames]
    return compute_distribution_vectors(all_detections)
```

---

### Step 3: Performance Estimation

Estimates mAP and latency of candidate partition plans **without executing them**.

**Key Equations:**

* Estimated mAP for block `p`:

  ```
  eAP_n,p = dot(AP_n, F_p)
  ```
* Estimated mAP for plan `κ`:

  ```
  eAP_κ = Σ (eAP_n,p * λ_p)
  ```

  where λ_p = object density of block `p`
* Estimated latency:

  ```
  eLat_κ = Σ (L_b_n) for each network used
  ```

**Pseudo-code:**

```python
def estimate_performance(network, block_dist, profiles):
    eAP = np.dot(profiles[network]['acc_vector'], block_dist)
    eLat = profiles[network]['latency']
    return eAP, eLat
```

---

### Step 4: Partition Planning

Recursively explores combinations of image partitions and network assignments using **dynamic programming + pruning**.

**Approach:**

* Enumerate splits recursively (Algorithm 1 in paper).
* Prune near-duplicate plans (within 1ms latency difference).
* Keep only plans with latency ≈ budget and high estimated accuracy.

**Pseudo-code (simplified):**

```python
def adaptive_partition(view, networks, profiles, budget):
    plans = []
    for n in networks:
        scaled_view = resize(view, n.input_size)
        dist_vec = estimate_distribution(scaled_view)
        eAP, eLat = estimate_performance(n, dist_vec, profiles)
        if eLat < 10_000:  # prune cutoff
            plans.append(Partition(view, n, eAP, eLat))
    # recursive subdivision
    for n in networks:
        subblocks = uniform_split(view, n.input_size)
        subplans = []
        for p in subblocks:
            smaller = smaller_networks(n)
            subplans += adaptive_partition(p, smaller, profiles, budget)
        merged = combine_plans(subplans)
        plans += prune(plans + merged)
    return prune(plans)
```

---

### Step 5: Partition Padding

Add minimal margins to prevent object truncation at partition borders using linear regression of object height/width vs image position.

**Pseudo-code:**

```python
def add_partition_padding(blocks, hist_detections):
    regressors = fit_height_width_vs_position(hist_detections)
    for b in blocks:
        pad = regressors.predict_padding(b.position)
        b.expand(pad)
    return blocks
```

---

## 2️⃣ Selective Execution Module (Online)

**Goal:** Dynamically execute subset of blocks under latency constraints.

### Step 1: Partition Selection (AIMD Controller)

Skips uninformative regions using previous detections.

**AIMD logic:**

* If no detections in block for `k` frames → increase skip window `w_p`.
* If detections reappear → reset `w_p` = 0.
* Decrease `w_p` by 1 per frame while skipped.

**Pseudo-code:**

```python
def partition_selection(blocks, prev_results):
    for p in blocks:
        if no_detections(p, prev_results):
            p.skip_window += 1
        else:
            p.skip_window = 0
        if p.skip_window > 0:
            p.skip_window -= 1
    return [p for p in blocks if p.skip_window == 0]
```

---

### Step 2: Batch Execution

Group blocks using the same NN and run inference in batch for GPU efficiency.

```python
def batch_execute(blocks):
    grouped = group_by_network(blocks)
    for net, batch in grouped.items():
        results = net.infer(batch)
        yield from results
```

NOTE: trtutils doesnt support batch inference or dynamic input sizes

---

### Step 3: Plan Control (PID Loop)

Adjust plan selection dynamically to maximize use of latency budget `T`.

**Controller:**

```
e(t) = T - L_actual
u(t) = Kp*e(t) + Ki*Σe(t) + Kd*(e(t) - e(t-1))
```

**Pseudo-code:**

```python
def plan_controller(pid, L_actual, T, plan_pool):
    e = T - L_actual
    pid.integrate(e)
    new_budget = pid.compute(e)
    return best_plan_under_budget(plan_pool, new_budget)
```

---

### Step 4: Bounding Box Merging

Merge per-block results using **Non-Maximum Suppression (NMS)** (IoU threshold = 0.5).

```python
final_results = nms(merge_block_detections(block_detections), iou_thresh=0.5)
```

---

## 🧠 Implementation Workflow

| Phase             | Module              | Runs On       | Key Artifacts                   |
| ----------------- | ------------------- | ------------- | ------------------------------- |
| Initialization    | Adaptive Partition  | Jetson / Edge | Partition Plan Pool             |
| Runtime           | Selective Execution | Edge Device   | Dynamic NN Scheduling           |
| Continuous Update | Re-profile          | Edge or Cloud | Updated plans when scene shifts |

---

## ⚙️ Developer Walkthrough Summary

1. **Offline Initialization**

   * Profile networks (latency + accuracy vectors)
   * Run oracle NN on few historical frames
   * Extract per-region object size distributions
   * Generate candidate partition plans (DP + pruning)
   * Store in `plan_pool.json`

2. **Online Execution Loop**

   * Load partition plan from pool (closest to latency budget)
   * For each incoming frame:

     * Apply selective execution (skip background regions)
     * Run batch inference per NN type
     * Adjust plan if actual latency < budget (PID loop)
     * Merge bounding boxes (NMS)
   * Periodically (e.g., daily): update object distribution → refresh plan pool

---

## 📦 End Result

Implementing this pipeline yields:

* 1.7×–8.1× **speedup** over uniform partitioning
* ≤0.2% **accuracy loss**
* Tunable latency that matches user-specified runtime constraints.

---

Would you like me to add **a visual block diagram (boxes/arrows)** and **pseudo-code structure following your `DetectorInterface` ABC pattern** (like for modular TRTEngine-based deployment)?
