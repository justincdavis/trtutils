#!/usr/bin/env bash

# get path of this file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# download yolo models
"$SCRIPT_DIR/download_yolo_models.sh"

# download rtdetr models
"$SCRIPT_DIR/download_rtdetr_models.sh"
