#!/usr/bin/env bash

set -euo pipefail

# Determine repo root (script is in <repo>/scripts)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
EXTERN_DIR="$REPO_DIR/extern"
DATA_DIR="$REPO_DIR/data"

# Ensure required directories exist
mkdir -p "$EXTERN_DIR"
mkdir -p "$DATA_DIR"/yolov7 "$DATA_DIR"/yolov8 "$DATA_DIR"/yolov9 "$DATA_DIR"/yolov10 "$DATA_DIR"/yolov11 "$DATA_DIR"/yolov12

cd "$EXTERN_DIR"

# export yolov7 (onnx)
git clone https://github.com/WongKinYiu/yolov7 || true
cd yolov7
python3 -m venv .venv
V7_PY="$(pwd)/.venv/bin/python"
"$V7_PY" -m pip install -U pip setuptools wheel
"$V7_PY" -m pip install -r requirements.txt
"$V7_PY" -m pip install onnx onnxruntime onnxslim onnxsim onnx_graphsurgeon
wget -nc https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
"$V7_PY" export.py --weights yolov7-tiny.pt --grid --end2end --simplify --img-size 640
mv -f yolov7-tiny.onnx "$DATA_DIR/yolov7/yolov7t.onnx"
cd ..

# export yolov8, yolov9, yolov10, yolov11 (onnx) via ultralytics
git clone https://github.com/ultralytics/ultralytics || true
cd ultralytics
python3 -m venv .venv
ULTRA_PY="$(pwd)/.venv/bin/python"
"$ULTRA_PY" -m pip install -U pip setuptools wheel
"$ULTRA_PY" -m pip install .
"$ULTRA_PY" -m pip install onnx onnxruntime onnxslim

# yolov8n.onnx
ULTRA_BIN="$(pwd)/.venv/bin"
"$ULTRA_BIN/yolo" export model=yolov8n.pt format=onnx simplify imgsz=640
mv -f yolov8n.onnx "$DATA_DIR/yolov8/yolov8n.onnx"

# yolov9t.onnx
"$ULTRA_BIN/yolo" export model=yolov9t.pt format=onnx simplify imgsz=640
mv -f yolov9t.onnx "$DATA_DIR/yolov9/yolov9t.onnx"

# yolov10n.onnx
"$ULTRA_BIN/yolo" export model=yolov10n.pt format=onnx simplify imgsz=640
mv -f yolov10n.onnx "$DATA_DIR/yolov10/yolov10n.onnx"

# yolo11n.onnx (rename to yolov11n.onnx at destination)
"$ULTRA_BIN/yolo" export model=yolo11n.pt format=onnx simplify imgsz=640
mv -f yolo11n.onnx "$DATA_DIR/yolov11/yolov11n.onnx"
cd ..

# export yolov12 (onnx)
git clone https://github.com/sunsmarterjie/yolov12 || true
cd yolov12
python3 -m venv .venv
Y12_BIN="$(pwd)/.venv/bin"
"$Y12_BIN/python" -m pip install -U pip setuptools wheel
"$Y12_BIN/python" -m pip install torch torchvision flash_attn timm albumentations onnx onnxruntime pycocotools pyyaml scipy onnxslim onnxruntime-gpu gradio opencv-python psutil huggingface-hub safetensors numpy supervision
"$Y12_BIN/python" -m pip install .
wget -nc https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12n.pt
"$Y12_BIN/yolo" export model=yolov12n.pt format=onnx simplify imgsz=640
mv -f yolov12n.onnx "$DATA_DIR/yolov12/yolov12n.onnx"
cd ..
