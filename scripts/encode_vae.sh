#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/media/datasets/yumi/hjh/Manipulation-Keyframes}"
CONFIG="${1:?Usage: $0 CONFIG MANIFEST OUT_DIR [NUM_GPUS]}"
MANIFEST="${2:?Usage: $0 CONFIG MANIFEST OUT_DIR [NUM_GPUS]}"
OUT_DIR="${3:?Usage: $0 CONFIG MANIFEST OUT_DIR [NUM_GPUS]}"
NUM_GPUS="${4:-${NUM_GPUS:-6}}"

mkdir -p "$OUT_DIR" "${PROJECT_ROOT}/logs/data"
cd "$PROJECT_ROOT"

LOG="${PROJECT_ROOT}/logs/data/encode_vae_$(date +%Y%m%d_%H%M%S).log"

torchrun \
  --standalone \
  --nproc_per_node="$NUM_GPUS" \
  keyframegen/data/encode_vae.py \
  --config "$CONFIG" \
  --manifest "$MANIFEST" \
  --out_dir "$OUT_DIR" \
  --batch_size "${BATCH_SIZE:-1}" \
  --tiled \
  2>&1 | tee "$LOG"
