---
name: accuracy-validation
description: Use when checking that a trtutils model's outputs match the original implementation - "validate accuracy", "compare against ultralytics/onnxruntime", "does the TRT output match the reference", "verify the new model family", after changing a postprocessor or preprocessor.
---

# Accuracy validation against a reference

Compare trtutils against the reference implementation running the same model file, so any
difference comes from trtutils (preprocessing, engine, postprocessing), not the weights.

## Steps

1. Pick the reference:
   - ultralytics models: `ultralytics.YOLO("<model>.pt").predict(img, imgsz=..., conf=...)`
     or `YOLO("<model>.onnx")` for the exact exported graph.
   - everything else: `onnxruntime.InferenceSession("<model>.onnx")` with the family's own
     preprocessing, decoded by the upstream postprocess code.
   Use the same thresholds (conf, NMS IoU, max detections) on both sides.

2. Use the tracked test images in `data/`: `horse.jpg`, `people.jpeg`, `street.jpg`,
   `cars.jpeg`. Add a domain image (e.g. an aerial photo for OBB) when the family needs one;
   do not commit it unless a test uses it.

3. Run trtutils from source (`PYTHONPATH=src`, log `trtutils.__file__`), with the engine
   built from that ONNX. Check fp32 first; then fp16 to see the precision loss.

4. Match and measure per image:
   - detections: greedy match by class + box IoU (>= 0.5); report reference count, trtutils
     count, matched count, mean/min box IoU of matches, max score difference
   - masks: mask IoU per matched pair (resize both to the original image size)
   - keypoints: mean / max pixel error over visible keypoints of matched pairs
   - classification: top-1 equality and top-5 overlap
   - text (OCR): exact string equality per region, plus a count of mismatches
   - depth: relative error after aligning scale where the model is relative-depth

5. Report a table per image and model: counts, matched, IoU/px error, and every unmatched
   detection with its score. Small differences near the confidence threshold are expected;
   explain any unmatched high-score detection before calling the port correct.

## Pitfalls seen before

- Input range: some exports want `(0, 1)`, others `(0, 255)`; a wrong `input_range`
  usually shows as zero or garbage detections, not a small drift.
- Size inputs (RT-DETR family): order and dtype come from the engine (`(w, h)`, int32/int64).
- TensorRT 8.6 can reorder engine outputs; outputs must be mapped by name.
- Some engines are nondeterministic (±1 px boxes); compare with tolerances, not equality.
- Letterbox vs plain resize changes boxes by a few pixels; match the reference's resize.
