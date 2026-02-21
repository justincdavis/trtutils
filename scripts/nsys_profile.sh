#!/usr/bin/env bash
# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# nsys_profile.sh - Comprehensive Nsight Systems profiling wrapper
# Targets: Python3, CUDA, NVTX, memory, kernel stats, GPU metrics
#
# Usage:
#   ./scripts/nsys_profile.sh [OPTIONS] -- [COMMAND...]
#
# Options:
#   -o, --output NAME    Output report name (default: profile_report)
#   -d, --duration SECS  Max capture duration in seconds (default: 60)
#   -g, --gpu-metrics    Enable GPU hardware metrics (requires permissions, see below)
#   -h, --help           Show this help
#
# GPU metrics note:
#   GPU metrics require setting /proc/sys/kernel/perf_event_paranoid:
#     sudo sh -c 'echo 2 > /proc/sys/kernel/perf_event_paranoid'
#   On WSL2 this may not be available. Use -g to opt in when permissions allow.
#
# Examples:
#   ./scripts/nsys_profile.sh -- python3 main.py --client-ip 127.0.0.1
#   ./scripts/nsys_profile.sh -o my_run -d 30 -- python3 main.py --client-ip 127.0.0.1
#   ./scripts/nsys_profile.sh -g -- python3 main.py --client-ip 127.0.0.1

set -euo pipefail

OUTPUT="profile_report"
DURATION=60
GPU_METRICS=()

usage() {
    sed -n '2,/^$/s/^# \?//p' "$0"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output)      OUTPUT="$2"; shift 2 ;;
        -d|--duration)    DURATION="$2"; shift 2 ;;
        -g|--gpu-metrics) GPU_METRICS=(--gpu-metrics-devices=all --gpu-metrics-frequency=10000); shift ;;
        -h|--help)        usage ;;
        --)               shift; break ;;
        *)                break ;;
    esac
done

if [[ $# -eq 0 ]]; then
    echo "Error: no command provided after --"
    echo "Usage: $0 [OPTIONS] -- [COMMAND...]"
    exit 1
fi

if ! command -v nsys &>/dev/null; then
    echo "Error: nsys not found. Install NVIDIA Nsight Systems or add it to PATH."
    exit 1
fi

exec nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    "${GPU_METRICS[@]}" \
    --sample=process-tree \
    --cpuctxsw=process-tree \
    --python-sampling=true \
    --python-backtrace=cuda \
    --duration="$DURATION" \
    --stats=true \
    --output="$OUTPUT" \
    --force-overwrite=true \
    "$@"
