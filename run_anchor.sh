#!/bin/bash
set -euo pipefail

GPU=${1:?Usage: $0 <gpu_id> <subject_dir>}
SUBJECT_DIR=${2:?Usage: $0 <gpu_id> <subject_dir>}
SEQ_NAME=$(basename "$SUBJECT_DIR")

CUDA_VISIBLE_DEVICES="$GPU" python -m inference run \
    --cfg configs/examples/presets/full.yaml \
    --footage "$(dirname "$SUBJECT_DIR")" \
    --seq_name "$SEQ_NAME" \
    --calib "$SUBJECT_DIR/calibration.json" \
    --out-tag thuman4 \
    -v
