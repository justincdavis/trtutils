#!/usr/bin/env bash

set -euo pipefail

# Determine repo root (script is in <repo>/scripts)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
EXTERN_DIR="$REPO_DIR/extern"
DATA_DIR="$REPO_DIR/data"

# Ensure required directories exist
mkdir -p "$EXTERN_DIR"
mkdir -p "$DATA_DIR"/rtdetrv1 "$DATA_DIR"/rtdetrv2 "$DATA_DIR"/rtdetrv3 "$DATA_DIR"/dfine

cd "$EXTERN_DIR"

git clone https://github.com/lyuwenyu/RT-DETR || true
cd RT-DETR
python3 -m venv .venv
RTDETR_PY="$(pwd)/.venv/bin/python"
"$RTDETR_PY" -m pip install -U pip setuptools wheel

# use sub directories for rtdetr v1 and v1
# export rtdetr v1
cd rtdetr_pytorch
"$RTDETR_PY" -m pip install -r requirements.txt onnxsim>=0.4
wget -nc https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth
"$RTDETR_PY" tools/export_onnx.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml -r rtdetr_r18vd_5x_coco_objects365_from_paddle.pth --check --simplify
mv -f model.onnx "$DATA_DIR/rtdetrv1/rtdetrv1_r18.onnx"
cd ..

# export rtdetr v2
cd rtdetrv2_pytorch
"$RTDETR_PY" -m pip install -r requirements.txt onnxsim>=0.4
wget -nc https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth
"$RTDETR_PY" tools/export_onnx.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml -r rtdetrv2_r18vd_120e_coco_rerun_48.1.pth --check --simplify
mv -f model.onnx "$DATA_DIR/rtdetrv2/rtdetrv2_r18.onnx"
cd ..

# done with rtdetr v1 and v2
cd ..

# export rtdetrv3 (onnx)
git clone https://github.com/clxia12/RT-DETRv3 || true
cd RT-DETRv3
python3 -m venv .venv
PY="$(pwd)/.venv/bin/python"
"$PY" -m pip install -U pip setuptools wheel
"$PY" -m pip install -r requirements.txt paddlepaddle==2.6.1 paddle2onnx==1.0.5 onnx==1.13.0 onnxsim>=0.4
# Download pretrained weights via Google Drive (bj.bcebos link returns 404)
"$PY" -m pip install gdown
mkdir -p weights
WEIGHTS_PATH="$(pwd)/weights/rtdetrv3_r18vd_6x_coco.pdparams"
if [ ! -f "$WEIGHTS_PATH" ]; then
    "$PY" -m gdown --id 1zIDOjn1qDccC3TBsDlGQHOjVrehd26bk -O "$WEIGHTS_PATH"
fi
"$PY" tools/export_model.py -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
              -o weights="$WEIGHTS_PATH" use_gpu=false trt=True \
              --output_dir=output_inference
PY_BIN="$(pwd)/.venv/bin"
"$PY_BIN/paddle2onnx" --model_dir=./output_inference/rtdetrv3_r18vd_6x_coco/ \
            --model_filename model.pdmodel  \
            --params_filename model.pdiparams \
            --opset_version 16 \
            --save_file rtdetrv3_r18vd_6x_coco.onnx
mv -f rtdetrv3_r18vd_6x_coco.onnx "$DATA_DIR/rtdetrv3/rtdetrv3_r18.onnx"
cd ..

# export d-fine (onnx)
git clone https://github.com/Peterande/D-FINE || true
cd D-FINE
python3 -m venv .venv
DFINE_PY="$(pwd)/.venv/bin/python"
"$DFINE_PY" -m pip install -U pip setuptools wheel
"$DFINE_PY" -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 torchvision==0.19.1
"$DFINE_PY" -m pip install -r requirements.txt onnx onnxsim --no-cache-dir
wget -nc https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth
# Inline patch: reduce exporter dummy batch size from 32 to 1 (idempotent)
sed -i -E 's/(data = torch\\.rand\\()[[:space:]]*[0-9]+,[[:space:]]*/\\11, /' tools/deployment/export_onnx.py || true
"$DFINE_PY" tools/deployment/export_onnx.py --check --simplify -c configs/dfine/dfine_hgnetv2_n_coco.yml -r dfine_n_coco.pth
mv -f dfine_n_coco.onnx "$DATA_DIR/dfine/dfine_n.onnx"
cd ..
