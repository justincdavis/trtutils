Here’s a **developer-oriented methodology summary and implementation walkthrough** of **FlexPatch**, breaking down its **dataflow, individual components, and operation pipeline** so an engineer could recreate the system from scratch.

---

## 🔶 Overview

**Goal:**
Enable **real-time and accurate object detection** on **high-resolution video (e.g., 1080p)** on mobile/edge devices by combining **tracking** and **selective detection**.

**Key Idea:**
Instead of running detection on every full frame (too slow), or relying entirely on tracking (accumulates error), **FlexPatch** detects only on small, dynamically chosen **patches** where tracking likely fails — efficiently packed into a small **patch cluster**.

---

## 🧭 Dataflow Summary

| Stage                 | Input                             | Output                                          | Purpose                                 |
| --------------------- | --------------------------------- | ----------------------------------------------- | --------------------------------------- |
| 1️⃣ Object Tracker    | Video frames, previous detections | Updated bounding boxes, optical flow features   | Continuously estimate object motion     |
| 2️⃣ Patch Recommender | Tracker output                    | List of patch candidates (with priority & type) | Identify regions needing re-detection   |
| 3️⃣ Patch Aggregator  | Candidate patches                 | Single packed “patch cluster”                   | Compact patches into one detection area |
| 4️⃣ Patched Detector  | Patch cluster                     | Fresh detections (bounding boxes)               | Re-detect objects in selected regions   |
| 5️⃣ Renderer          | Detections + tracked objects      | On-screen output                                | Visual display, feeds back into tracker |

Data moves cyclically — each new detection refines the tracker, which in turn guides the next patch selection.

---

## 🧩 Component Walkthrough

### 1. Continuous Object Tracker

**Purpose:** Maintain object positions between detector runs.

**Steps:**

1. For each frame, compute **optical flow** (e.g., ORB features + Lucas-Kanade flow).
2. **Propagate** detected bounding boxes:

   * **Detection Propagation:** When a new detection arrives, map detections forward.
   * **Successive Propagation:** While waiting for new detection, track frame-to-frame.
3. Apply **Incremental Detection Propagation (IDP):** Cache intermittent frames to improve propagation accuracy.
4. Use **association-based smoothing** — compare IoU with previous boxes; remove only after several missed matches.

**Implementation Notes:**

* Optical flow features extracted using OpenCV/JavaCV.
* Bounding box aging & IoU thresholds configurable.

---

### 2. Patch Recommender

**Purpose:** Choose subregions where detection is most valuable.

It generates two patch types:

* **Tracking-failure patches** — areas where current tracking likely went wrong.
* **New-object patches** — areas where new objects might have appeared.

---

#### (a) Tracking-Failure Patch Recommendation

**Logic:**

1. Extract features for each tracked box:

   * Minimum eigenvalue (feature point quality)
   * NCC (appearance consistency)
   * Bounding box acceleration (occlusion indicator)
   * Std. dev. of optical flow (motion inconsistency)
   * Detector confidence score
2. Feed features into a **Decision Tree Classifier** (trained offline using IoU labels):

   * Classify as High / Medium / Low priority
3. Add **padding** (equal to box size) to account for tracking drift.
4. Increase priority if not re-detected for several frames.

**Output:** List of candidate patches (bounding boxes + padding + priority).

---

#### (b) New-Object Patch Recommendation

**Logic:**

1. Divide frame into uniform **cells** (e.g., 8×8 pixels).
2. Compute two metrics per cell:

   * **Edge Intensity (EI):** via Canny edge detector → signals potential new object.
   * **Refresh Interval (RI):** frames since last detection in this region.
3. Compute cell priority:

   ```
   priority = min(50, RI) + W × (EI > T)
   ```

   * W, T are tunable constants.
4. Group adjacent high-priority cells into larger patches (e.g., 20×22 cells).
5. Skip cells overlapping existing bounding boxes (those handled by tracker).

**Output:** Candidate patches for unseen objects with associated priorities.

---

### 3. Patch Aggregator

**Purpose:** Efficiently fit prioritized patches into one rectangular patch cluster for detection.

#### (a) Aggregation Policy

* **Cluster Size:** Typically 640×360 for 1080p video. Trade-off:

  * Smaller cluster → faster detection, fewer patches.
  * Larger cluster → slower detection, more patches.
* **Patch Type Ratio:** Alternate between clusters dominated by:

  * Tracking-failure patches
  * New-object patches
    (Default 3:1 ratio; adjustable per dataset)

#### (b) Aggregation Algorithm

Implements **Guillotine 2D Bin Packing**:

1. Sort patches by priority.
2. Maintain list of free rectangles (`F`).
3. Iteratively fit each patch into a free rectangle.
4. After placing, **split** remaining space along shorter axis.
5. Downsample oversized patches slightly if needed.

**Output:**
Patch cluster image, with each patch’s coordinates and type (for reconstruction).

**Latency:** ~2 ms average per packing cycle on mobile CPU.

---

### 4. Patched Object Detector

**Purpose:** Run detection on reduced, high-value image area.

**Process:**

1. Crop and merge selected patches into the patch cluster.
2. Resize cluster to detector input resolution (e.g., 640×360).
3. Run lightweight detector (Tiny YOLOv4 or similar).
4. Map resulting detections back to full-frame coordinates.
5. Feed fresh detections to tracker and renderer.

**Latency Example:**

| Resolution | Latency (ms) |
| ---------- | ------------ |
| 1080p      | 1029         |
| 640×360    | 139          |

---

### 5. Renderer

**Purpose:** Display output and manage synchronization.

* Renders both tracked and newly detected boxes.
* Provides visual verification.
* Updates tracker states with new detections.

---

## ⚙️ Implementation Stack

| Module            | Suggested Technology                                                   |
| ----------------- | ---------------------------------------------------------------------- |
| Object Tracker    | OpenCV/JavaCV (ORB + Lucas-Kanade)                                     |
| Patch Recommender | Python (scikit-learn for decision tree) / Java for real-time inference |
| Patch Aggregator  | Custom C++ or Java algorithm (Guillotine bin packing)                  |
| Detector          | Tiny YOLOv4 (Darknet → TensorFlow Lite)                                |
| Mobile Framework  | Android SDK, TensorFlow Lite GPU delegate                              |

---

## 🔁 Full System Flow

```
for each frame in video_stream:
    tracked_objects, features = tracker.update(frame)
    patches_TF = tracking_failure_recommender(tracked_objects, features)
    patches_NO = new_object_recommender(frame, last_detection_map)
    patch_cluster = aggregator.pack(patches_TF + patches_NO)
    detections = detector.run(patch_cluster)
    tracker.refresh(detections)
    renderer.display(detections, frame)
```

---

## ✅ Expected Outcome

By implementing each component as described:

* Detection latency drops from ~1 s → ~139 ms per cycle.
* Accuracy improves up to **146% AP** over standard DBT frameworks.
* Power use reduces to ~37% of baseline continuous detection.

---

Would you like me to produce a **block diagram or pseudo-code representation** of this full pipeline next (e.g., for documentation or implementation reference)?
