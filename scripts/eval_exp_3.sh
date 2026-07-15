#!/usr/bin/env bash
set -euo pipefail

cd /media/datasets/yumi/hjh/Manipulation-Keyframes
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /media/datasets/yumi/hjh/conda_envs/keyframe

export DIFFSYNTH_SKIP_DOWNLOAD=True
export DIFFSYNTH_MODEL_BASE_PATH=/media/datasets/yumi/hjh/Manipulation-Keyframes/models
export TOKENIZERS_PARALLELISM=false

PYTHON=/media/datasets/yumi/hjh/conda_envs/keyframe/bin/python
CONFIG=configs/agibot/infer_eval/exp3_dual_context_stable_lora_rank128.json
RUN_DIR=runs/exp3_dual_context_stable_lora_rank128

run_steps() {
  local gpu="$1"
  shift

  for step in "$@"; do
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u keyframegen/infer/infer_exp.py \
      --config "${CONFIG}" \
      --step "${step}" \
      > "${RUN_DIR}/eval_step_$(printf "%06d" "${step}").log" 2>&1
  done
}

run_steps 4 9000 7000 5000 3000 1000 &
run_steps 5 8500 6500 4500 2500 500 &
run_steps 6 8000 6000 4000 2000 &
run_steps 7 7500 5500 3500 1500 &

wait
