# Webcam Demos

Three task-oriented webcam demos that accept any supported model by name. Each
script auto-downloads the ONNX, builds a TensorRT engine, and runs the live
inference loop.

## Demos

| Script                | Task                  | Example model names                              |
| --------------------- | --------------------- | ------------------------------------------------ |
| `detector.py`         | Object detection      | `yolov10n`, `rtdetrv2_r18`, `dfine_n`, `rfdetr_n` |
| `classifier.py`       | Image classification  | `resnet18`, `vit_b_16`, `efficientnet_b0`        |
| `depth_estimator.py`  | Monocular depth       | `depth_anything_v2_small`, `depth_anything_v2_base` |

Any name accepted by `trtutils.models._utils.get_valid_models()` for the
relevant task family is valid. Unknown names produce a clean `ValueError`
listing all valid options.

## Requirements

- NVIDIA GPU with CUDA + TensorRT.
- `trtutils` installed with the appropriate CUDA extra (e.g. `pip install trtutils[cu12]`).
- Webcam or video device.
- Demo deps:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python detector.py --model yolov10n
python detector.py --model rtdetrv2_r18 --source 0
python classifier.py --model resnet18 --top-k 5
python depth_estimator.py --model depth_anything_v2_small
```

Each demo caches the resulting `<name>.onnx` and `<name>.engine` under
`demos/webcam/data/`. Delete those files to force a rebuild.

Press `q` or close the window to exit.

## Output

- **`detector.py`** — Bounding boxes drawn on the live feed with running FPS.
- **`classifier.py`** — Top-K ImageNet labels overlaid on the live feed with running FPS.
- **`depth_estimator.py`** — Side-by-side webcam frame + `COLORMAP_INFERNO`
  depth visualization with running FPS.
