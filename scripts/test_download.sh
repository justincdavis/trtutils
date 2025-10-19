#!/bin/bash

set -uo pipefail

TMP_DIR=$(mktemp -d -t trt_download.XXXXXX)
printf 'Using temp directory: %s\n\n' "$TMP_DIR"

MODELS=(
  "yolov7t:yolov7t.onnx"
  "yolov8n:yolov8n.onnx"
  "yolov9t:yolov9t.onnx"
  "yolov10n:yolov10n.onnx"
  "yolov11n:yolov11n.onnx"
  "yolov12n:yolov12n.onnx"
  "yolov13n:yolov13n.onnx"
  "yoloxn:yoloxn.onnx"
  "rtdetrv1_r18:rtdetrv1_r18.onnx"
  "rtdetrv2_r18:rtdetrv2_r18.onnx"
  "rtdetrv3_r18:rtdetrv3_r18.onnx"
  "dfine_n:dfine_n.onnx"
  "deim_dfine_n:deim_dfine_n.onnx"
  "deimv2_atto:deimv2_atto.onnx"
  "rfdetr_n:rfdetr_n.onnx"
)

successes=()
failures=()

for entry in "${MODELS[@]}"; do
  IFS=":" read -r model output <<< "$entry"

  output_path="$TMP_DIR/$output"
  rm -f "$output_path"

  printf 'Downloading %s -> %s...\n' "$model" "$output_path"

  python3 -m trtutils download --model "$model" --output "$output_path" --accept || true

  if [[ -f "$output_path" ]]; then
    successes+=("$model:$output_path")
    printf '[  OK  ] %s (created %s)\n\n' "$model" "$output_path"
  else
    failures+=("$model")
    printf '[ FAIL ] %s (missing %s)\n\n' "$model" "$output_path"
  fi
done

printf 'Summary:\n'

printf '  Successful downloads (%d):\n' "${#successes[@]}"
for entry in "${successes[@]}"; do
  IFS=":" read -r model output_path <<< "$entry"
  printf '    - %s -> %s\n' "$model" "$output_path"
done

if (( ${#failures[@]} > 0 )); then
  printf '  Failed downloads (%d):\n' "${#failures[@]}"
  for model in "${failures[@]}"; do
    printf '    - %s\n' "$model"
  done
else
  printf '  Failed downloads: none\n'
fi

exit ${#failures[@]}
